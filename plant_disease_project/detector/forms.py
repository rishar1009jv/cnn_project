"""
detector/forms.py
Image upload form with server-side validation.
"""
from django import forms
from django.conf import settings
from pathlib import Path


class LeafUploadForm(forms.Form):
    image = forms.ImageField(
        label="Select Leaf Image",
        help_text="Accepted formats: JPG, JPEG, PNG. Maximum size: 10 MB.",
    )

    def clean_image(self):
        image = self.cleaned_data.get("image")
        if image is None:
            raise forms.ValidationError("No image file was uploaded.")

        # ── Extension check ───────────────────────────────────────────────
        ext = Path(image.name).suffix.lower()
        allowed = getattr(settings, "ALLOWED_IMAGE_EXTENSIONS", [".jpg", ".jpeg", ".png"])
        if ext not in allowed:
            raise forms.ValidationError(
                f"Unsupported file type '{ext}'. "
                f"Please upload one of: {', '.join(allowed)}."
            )

        # ── Size check ────────────────────────────────────────────────────
        max_mb = getattr(settings, "MAX_UPLOAD_SIZE_MB", 10)
        max_bytes = max_mb * 1024 * 1024
        if image.size > max_bytes:
            raise forms.ValidationError(
                f"File too large ({image.size / 1024 / 1024:.1f} MB). "
                f"Maximum allowed size is {max_mb} MB."
            )

        return image
