"""
detector/ml/preprocessing.py

Image quality assessment and enhancement pipeline.

Design decisions
----------------
* Blur detection uses the variance of the Laplacian operator (a standard,
  fast estimator).  A low variance indicates few sharp edges → likely blurry.
  This is an *estimate* – it cannot guarantee whether disease features are
  preserved or lost.
* Enhancement is conservative:
  - CLAHE (Contrast Limited Adaptive Histogram Equalisation) on the L channel
    of LAB colour space for contrast.  Limits are kept tight to avoid artefacts.
  - Gentle Gaussian denoising only when noise is detected.
  - No aggressive sharpening or deconvolution that could introduce artefacts.
* Enhancement is skipped when the image is already of good quality.
* If the quality score is below UNRECOVERABLE_THRESHOLD the caller is advised
  to ask for a clearer photo; enhancement is not attempted.
* The original PIL image is never modified in place.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# ── Tunable thresholds ──────────────────────────────────────────────────────
BLUR_THRESHOLD = 100.0        # Laplacian variance below this → blurry
UNRECOVERABLE_THRESHOLD = 15.0  # Below this → too poor even to try
NOISE_THRESHOLD = 25.0         # Std-dev estimate above this → noisy


@dataclass
class QualityReport:
    """Results of the quality assessment step."""
    laplacian_variance: float = 0.0
    is_blurred: bool = False
    is_unrecoverable: bool = False
    noise_estimate: float = 0.0
    width: int = 0
    height: int = 0
    channels: int = 3
    quality_label: str = "Unknown"
    quality_score: float = 0.0   # 0–100 normalised convenience score
    messages: list[str] = field(default_factory=list)


def assess_quality(pil_image: Image.Image) -> QualityReport:
    """
    Analyse image quality and return a :class:`QualityReport`.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        The image to assess (any mode – converted internally).

    Returns
    -------
    QualityReport
    """
    img_rgb = pil_image.convert("RGB")
    arr = np.array(img_rgb)

    # Convert to grayscale for Laplacian
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Noise estimate: high-pass residual std dev
    blurred_for_noise = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_est = float(np.std(gray.astype(np.float32) - blurred_for_noise.astype(np.float32)))

    h, w = gray.shape
    channels = len(img_rgb.getbands())

    is_unrecoverable = lap_var < UNRECOVERABLE_THRESHOLD
    is_blurred = lap_var < BLUR_THRESHOLD

    # Normalised convenience score (clipped 0–100)
    quality_score = min(100.0, (lap_var / BLUR_THRESHOLD) * 100.0)

    if is_unrecoverable:
        label = "Very Poor"
        msgs = [
            "The image is extremely blurry or low-contrast.",
            "Please retake the photo in good lighting, holding the camera steady.",
        ]
    elif is_blurred:
        label = "Low"
        msgs = [
            "The image appears blurry. Enhancement will be attempted.",
            "Note: blur detection is an estimate; disease features may still "
            "be present or absent regardless of this score.",
        ]
    elif quality_score < 80:
        label = "Acceptable"
        msgs = ["Image quality is acceptable for classification."]
    else:
        label = "Good"
        msgs = ["Image quality is good."]

    return QualityReport(
        laplacian_variance=lap_var,
        is_blurred=is_blurred,
        is_unrecoverable=is_unrecoverable,
        noise_estimate=noise_est,
        width=w,
        height=h,
        channels=channels,
        quality_label=label,
        quality_score=round(quality_score, 2),
        messages=msgs,
    )


def enhance_image(pil_image: Image.Image, report: QualityReport) -> Tuple[Image.Image, bool]:
    """
    Apply conservative enhancement when appropriate.

    Enhancement is skipped if:
    * The image quality is already "Good" or "Acceptable".
    * The image is unrecoverable (quality too poor – ask for retake instead).

    Returns
    -------
    (enhanced_pil_image, enhancement_was_applied)
    """
    if report.is_unrecoverable or not report.is_blurred:
        # Nothing to do – return a copy of the original
        return pil_image.copy(), False

    logger.info("Applying image enhancement (Laplacian variance=%.2f)", report.laplacian_variance)
    arr = np.array(pil_image.convert("RGB"))

    # ── Step 1: CLAHE on LAB L-channel for contrast ─────────────────────
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    arr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # ── Step 2: Gentle denoising only when noise is high ────────────────
    if report.noise_estimate > NOISE_THRESHOLD:
        arr = cv2.fastNlMeansDenoisingColored(arr, None, h=5, hColor=5,
                                               templateWindowSize=7,
                                               searchWindowSize=21)

    # ── Step 3: Mild unsharp mask to recover some edge detail ───────────
    pil_enhanced = Image.fromarray(arr)
    pil_enhanced = pil_enhanced.filter(ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=3))

    return pil_enhanced, True


def preprocess_for_model(pil_image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize and normalise an image to the format expected by the CNN.

    Returns
    -------
    numpy.ndarray of shape (1, H, W, 3) with float32 values in [0, 1].
    """
    img = pil_image.convert("RGB").resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def pil_to_bytes(pil_image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert a PIL image to raw bytes (for saving or serving)."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return buf.getvalue()
