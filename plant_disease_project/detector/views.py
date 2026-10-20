"""
detector/views.py

All HTTP views for the plant disease detection application.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from PIL import Image, UnidentifiedImageError

from .forms import LeafUploadForm
from .ml.predict import ModelNotReadyError, predict_disease
from .ml.preprocessing import assess_quality, enhance_image, pil_to_bytes

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(settings.MODEL_PATH)
LABELS_PATH = Path(settings.CLASS_LABELS_PATH)
RECOMMENDATIONS_PATH = Path(settings.RECOMMENDATIONS_PATH)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_recommendations() -> dict:
    """Load disease recommendations JSON. Returns {} on error."""
    try:
        with open(RECOMMENDATIONS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Remove internal metadata key
        data.pop("_metadata", None)
        return data
    except Exception as exc:
        logger.warning("Could not load recommendations: %s", exc)
        return {}


def _get_recommendation(class_name: str, recs: dict) -> dict | None:
    """Return recommendation dict for class_name, with normalized fuzzy fallback."""
    if not class_name or not recs:
        return None
    if class_name in recs:
        return recs[class_name]
    
    # Normalize alphanumeric characters to match slight differences in formatting
    def _norm(s: str) -> str:
        return "".join(c.lower() for c in s if c.isalnum())

    target = _norm(class_name)
    for k, v in recs.items():
        if _norm(k) == target:
            return v
    return None


def _save_media_image(pil_image: Image.Image, prefix: str) -> str:
    """
    Save a PIL image to MEDIA_ROOT and return its relative URL path.
    """
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.jpg"
    rel_path = f"predictions/{filename}"
    img_bytes = pil_to_bytes(pil_image, fmt="JPEG")
    saved_name = default_storage.save(rel_path, ContentFile(img_bytes))
    return settings.MEDIA_URL + saved_name


def _is_model_available() -> bool:
    return MODEL_PATH.exists() and LABELS_PATH.exists()


# ── Views ─────────────────────────────────────────────────────────────────────

def home(request):
    """Home / landing page."""
    context = {
        "model_ready": _is_model_available(),
        "supported_plants": [
            {"name": "Apple", "diseases": ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"]},
            {"name": "Corn", "diseases": ["Gray Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy"]},
            {"name": "Grape", "diseases": ["Black Rot", "Esca", "Leaf Blight", "Healthy"]},
            {"name": "Tomato", "diseases": ["Early Blight", "Late Blight", "Leaf Mold", "Septoria", "TYLCV", "Mosaic Virus", "Healthy"]},
            {"name": "Potato", "diseases": ["Early Blight", "Late Blight", "Healthy"]},
            {"name": "Pepper", "diseases": ["Bacterial Spot", "Healthy"]},
            {"name": "Peach", "diseases": ["Bacterial Spot", "Healthy"]},
            {"name": "Cherry", "diseases": ["Powdery Mildew", "Healthy"]},
            {"name": "Strawberry", "diseases": ["Leaf Scorch", "Healthy"]},
        ],
    }
    return render(request, "detector/home.html", context)


@require_http_methods(["GET", "POST"])
def upload_and_predict(request):
    """
    GET  → render upload form
    POST → validate, assess quality, optionally enhance, run inference, show results
    """
    form = LeafUploadForm()
    context = {"form": form}

    if request.method == "POST":
        form = LeafUploadForm(request.POST, request.FILES)

        if not form.is_valid():
            context["form"] = form
            return render(request, "detector/upload.html", context)

        # ── Read uploaded image ────────────────────────────────────────────
        uploaded_file = form.cleaned_data["image"]
        try:
            original_pil = Image.open(uploaded_file).convert("RGB")
        except UnidentifiedImageError:
            form.add_error("image", "The uploaded file could not be read as an image. Please try a different file.")
            context["form"] = form
            return render(request, "detector/upload.html", context)

        # ── Quality assessment ─────────────────────────────────────────────
        quality_report = assess_quality(original_pil)

        # ── Save original image for display ───────────────────────────────
        original_url = _save_media_image(original_pil, "original")

        # ── Enhancement (if needed) ────────────────────────────────────────
        enhanced_pil, enhancement_applied = enhance_image(original_pil, quality_report)
        enhanced_url = None
        if enhancement_applied:
            enhanced_url = _save_media_image(enhanced_pil, "enhanced")

        # ── Model check ───────────────────────────────────────────────────
        if not _is_model_available():
            context.update({
                "form": form,
                "quality_report": quality_report,
                "original_url": original_url,
                "enhanced_url": enhanced_url,
                "enhancement_applied": enhancement_applied,
                "model_missing": True,
                "model_path": str(MODEL_PATH),
                "labels_path": str(LABELS_PATH),
            })
            return render(request, "detector/results.html", context)

        # ── If image quality is too poor, warn and stop ──────────────────
        if quality_report.is_unrecoverable:
            context.update({
                "form": form,
                "quality_report": quality_report,
                "original_url": original_url,
                "enhancement_applied": False,
                "unrecoverable_quality": True,
            })
            return render(request, "detector/results.html", context)

        # ── Inference ─────────────────────────────────────────────────────
        # Use enhanced image when available, otherwise original
        inference_image = enhanced_pil if enhancement_applied else original_pil
        try:
            prediction = predict_disease(
                pil_image=inference_image,
                model_path=MODEL_PATH,
                labels_path=LABELS_PATH,
                input_size=settings.CNN_INPUT_SIZE,
            )
        except ModelNotReadyError as exc:
            context.update({
                "form": form,
                "model_missing": True,
                "model_error": str(exc),
                "original_url": original_url,
                "quality_report": quality_report,
            })
            return render(request, "detector/results.html", context)
        except Exception as exc:
            logger.exception("Unexpected error during inference")
            context.update({
                "form": form,
                "inference_error": f"An unexpected error occurred during prediction: {exc}",
                "original_url": original_url,
                "quality_report": quality_report,
            })
            return render(request, "detector/results.html", context)

        # ── Load recommendations ──────────────────────────────────────────
        all_recs = _load_recommendations()
        recommendation = _get_recommendation(prediction.predicted_class, all_recs)

        # ── Optionally persist prediction to DB ───────────────────────────
        try:
            from .models import PredictionRecord
            PredictionRecord.objects.create(
                predicted_class=prediction.predicted_class,
                confidence_score=prediction.confidence,
                image_quality_score=quality_report.quality_score,
                is_blurred=quality_report.is_blurred,
                enhancement_applied=enhancement_applied,
            )
        except Exception as exc:
            logger.warning("Could not save prediction record: %s", exc)

        context.update({
            "quality_report": quality_report,
            "original_url": original_url,
            "enhanced_url": enhanced_url,
            "enhancement_applied": enhancement_applied,
            "prediction": prediction,
            "recommendation": recommendation,
            "confidence_percent": round(prediction.confidence * 100, 2),
        })
        return render(request, "detector/results.html", context)

    return render(request, "detector/upload.html", context)


def history(request):
    """View prediction history."""
    from .models import PredictionRecord
    records = PredictionRecord.objects.all()[:50]
    return render(request, "detector/history.html", {"records": records})
