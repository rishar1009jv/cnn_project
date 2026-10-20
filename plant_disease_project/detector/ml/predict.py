"""
detector/ml/predict.py

Inference module – loads the saved Keras model once at startup and
exposes a single function: predict_disease().

Important notes
---------------
* The model is loaded lazily on the first call and cached globally.
  This avoids re-loading the large file on every HTTP request.
* Confidence scores are model output probabilities.  High confidence
  does NOT guarantee a correct prediction; the system is experimental.
* If the model or class-label file is missing, a clear ModelNotReadyError
  is raised – never a silent fallback to a fake result.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Module-level singletons ─────────────────────────────────────────────────
_model = None
_class_labels: Optional[Dict[int, str]] = None
_load_lock = threading.Lock()


class ModelNotReadyError(RuntimeError):
    """Raised when the saved model or class-label file is missing."""


@dataclass
class PredictionResult:
    predicted_class: str          # e.g. "Tomato___Early_blight"
    predicted_label: str          # Human-friendly label, e.g. "Tomato – Early Blight"
    confidence: float             # 0.0–1.0
    top_k: List[Tuple[str, float]]  # [(label, confidence), ...] – top-5
    model_ready: bool = True


def _load_model_and_labels(model_path: Path, labels_path: Path) -> None:
    """Load Keras model and class-label mapping into module globals."""
    global _model, _class_labels

    if not model_path.exists():
        raise ModelNotReadyError(
            f"Trained model not found at: {model_path}\n"
            "Please run the training script first:\n"
            "  python detector/ml/train.py"
        )
    if not labels_path.exists():
        raise ModelNotReadyError(
            f"Class-label file not found at: {labels_path}\n"
            "It is generated automatically during training."
        )

    # Import TF here so Django can import this module before TF is available
    import tensorflow as tf  # noqa: PLC0415

    logger.info("Loading model from %s …", model_path)
    _model = tf.keras.models.load_model(str(model_path))
    logger.info("Model loaded successfully.")

    with open(labels_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    # class_labels.json may be {"0": "Apple___Apple_scab", ...}
    _class_labels = {int(k): v for k, v in raw.items()}
    logger.info("Loaded %d class labels.", len(_class_labels))


def _ensure_loaded(model_path: Path, labels_path: Path) -> None:
    """Thread-safe lazy initialisation."""
    global _model, _class_labels
    if _model is not None and _class_labels is not None:
        return
    with _load_lock:
        if _model is None or _class_labels is None:
            _load_model_and_labels(model_path, labels_path)


def _make_human_label(raw_class: str) -> str:
    """
    Convert a PlantVillage-style class name to a readable label.
    'Tomato___Early_blight'  →  'Tomato – Early Blight'
    'Apple___healthy'        →  'Apple – Healthy'
    """
    parts = raw_class.replace("___", "|||").replace("__", " ").split("|||")
    if len(parts) == 2:
        plant, disease = parts
        disease = disease.replace("_", " ").title()
        return f"{plant.strip()} – {disease.strip()}"
    return raw_class.replace("_", " ").title()


def predict_disease(
    pil_image: Image.Image,
    model_path: Path,
    labels_path: Path,
    input_size: Tuple[int, int] = (224, 224),
    top_k: int = 5,
) -> PredictionResult:
    """
    Run inference on a PIL image.

    Parameters
    ----------
    pil_image   : PIL image (any mode – converted internally).
    model_path  : Path to the saved .h5 model.
    labels_path : Path to class_labels.json.
    input_size  : (height, width) the model was trained on.
    top_k       : Number of top predictions to return.

    Returns
    -------
    PredictionResult

    Raises
    ------
    ModelNotReadyError  If the model or label file is missing.
    """
    _ensure_loaded(model_path, labels_path)

    # Preprocess
    img = pil_image.convert("RGB").resize(input_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)   # shape (1, H, W, 3)

    # Inference
    predictions = _model.predict(batch, verbose=0)[0]   # shape (num_classes,)

    # Top-k
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_pairs = [
        (_make_human_label(_class_labels[int(i)]), float(predictions[i]))
        for i in top_indices
    ]

    best_idx = int(top_indices[0])
    best_class = _class_labels[best_idx]
    best_label = _make_human_label(best_class)
    best_conf = float(predictions[best_idx])

    logger.info("Prediction: %s (%.1f%%)", best_label, best_conf * 100)

    return PredictionResult(
        predicted_class=best_class,
        predicted_label=best_label,
        confidence=best_conf,
        top_k=top_pairs,
    )
