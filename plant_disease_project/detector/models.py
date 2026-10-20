"""
detector/models.py
Django database models for storing prediction history.
"""
from django.db import models
from django.utils import timezone


class PredictionRecord(models.Model):
    """Stores each prediction event for audit and history purposes."""

    original_image = models.ImageField(upload_to='predictions/original/', blank=True)
    enhanced_image = models.ImageField(upload_to='predictions/enhanced/', blank=True, null=True)

    predicted_class = models.CharField(max_length=200)
    confidence_score = models.FloatField()  # 0.0–1.0

    image_quality_score = models.FloatField(null=True, blank=True)
    is_blurred = models.BooleanField(default=False)
    enhancement_applied = models.BooleanField(default=False)

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Prediction Record'
        verbose_name_plural = 'Prediction Records'

    def __str__(self):
        return f"{self.predicted_class} ({self.confidence_score:.1%}) @ {self.created_at:%Y-%m-%d %H:%M}"

    def confidence_percent(self):
        return round(self.confidence_score * 100, 2)

    @property
    def human_label(self):
        parts = self.predicted_class.replace("___", "|||").replace("__", " ").split("|||")
        if len(parts) == 2:
            plant, disease = parts
            disease = disease.replace("_", " ").title()
            return f"{plant.strip()} – {disease.strip()}"
        return self.predicted_class.replace("_", " ").title()
