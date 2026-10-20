"""
detector/apps.py
Django AppConfig for the detector application.
"""
from django.apps import AppConfig


class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'
    verbose_name = 'Plant Disease Detector'
