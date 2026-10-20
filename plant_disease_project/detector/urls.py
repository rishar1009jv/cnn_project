"""detector/urls.py – URL patterns for the detector app."""
from django.urls import path
from . import views

app_name = "detector"

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload_and_predict, name="upload"),
    path("history/", views.history, name="history"),
]
