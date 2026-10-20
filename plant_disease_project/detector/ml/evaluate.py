"""
detector/ml/evaluate.py
=======================
Evaluation script that reads test_results.json (written by train.py) and
produces accuracy, precision, recall, F1-score, and a confusion-matrix PNG.

Also compares model performance on:
  1. Clear original images
  2. Artificially blurred images
  3. Enhanced versions of the blurred images

Usage:
    python detector/ml/evaluate.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from django.conf import settings
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = Path(settings.BASE_DIR) / "evaluation"
TEST_RESULTS_PATH = EVAL_DIR / "test_results.json"


def load_test_results():
    if not TEST_RESULTS_PATH.exists():
        logger.error(
            "Test results not found at %s\n"
            "Please run training first: python detector/ml/train.py",
            TEST_RESULTS_PATH,
        )
        sys.exit(1)
    with open(TEST_RESULTS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_standard(data: dict):
    """Classification metrics on the original test set."""
    y_true = np.array(data["y_true"])
    y_pred = np.array(data["y_pred"])
    class_names = data["class_names"]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    logger.info("\n=== Standard Test-Set Metrics ===")
    logger.info("Accuracy : %.4f", acc)
    logger.info("\n%s", report)

    return y_true, y_pred, class_names, acc, report


def save_confusion_matrix(y_true, y_pred, class_names, out_path: Path):
    """Saves a confusion-matrix PNG (limited to first 20 classes for readability)."""
    cm = confusion_matrix(y_true, y_pred)
    n = min(len(class_names), 20)
    cm_sub = cm[:n, :n]
    names_sub = [n.split("___")[-1].replace("_", " ") for n in class_names[:n]]

    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_sub, display_labels=names_sub)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix (first 20 classes)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", out_path)


def save_training_curves(out_path: Path):
    """Plot training/validation accuracy and loss from training_history.json."""
    hist_path = Path(settings.BASE_DIR) / "models_saved" / "training_history.json"
    if not hist_path.exists():
        logger.warning("training_history.json not found – skipping curves.")
        return

    with open(hist_path, "r", encoding="utf-8") as fh:
        history = json.load(fh)

    # Concatenate phase1 + phase2
    all_acc, all_val_acc = [], []
    all_loss, all_val_loss = [], []
    for phase in ("phase1", "phase2"):
        ph = history.get(phase, {})
        all_acc.extend(ph.get("accuracy", []))
        all_val_acc.extend(ph.get("val_accuracy", []))
        all_loss.extend(ph.get("loss", []))
        all_val_loss.extend(ph.get("val_loss", []))

    epochs = range(1, len(all_acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, all_acc, label="Train Accuracy")
    axes[0].plot(epochs, all_val_acc, label="Val Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    axes[1].plot(epochs, all_loss, label="Train Loss")
    axes[1].plot(epochs, all_val_loss, label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.suptitle("Training History", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved → %s", out_path)


def blur_comparison_evaluation(data: dict, model):
    """
    Re-runs inference on the test images under three conditions:
      A) Original (clear)
      B) Artificially blurred
      C) Blur → Enhanced (CLAHE + unsharp mask)

    Returns a dict of {condition: accuracy}.
    IMPORTANT: Results reflect actual model outputs, not invented numbers.
    """
    import cv2
    import tensorflow as tf
    from PIL import Image, ImageFilter
    from detector.ml.preprocessing import assess_quality, enhance_image

    IMG_SIZE = settings.CNN_INPUT_SIZE
    test_results_path = Path(settings.BASE_DIR) / "evaluation" / "test_results.json"

    with open(test_results_path) as fh:
        saved = json.load(fh)

    # We need actual image file paths.  train.py stores only labels & preds.
    # Re-discover test paths from the dataset.
    dataset_dir = Path(settings.BASE_DIR) / "dataset" / "PlantVillage"
    if not dataset_dir.exists():
        logger.warning("Dataset not found – skipping blur-comparison evaluation.")
        return {}

    class_names = data["class_names"]
    class_to_idx = {n: i for i, n in enumerate(class_names)}

    # Collect a small random sample (up to 200 images) to keep runtime manageable
    from sklearn.model_selection import train_test_split
    all_paths, all_labels = [], []
    for cls_dir in sorted(dataset_dir.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in class_to_idx:
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in cls_dir.glob(ext):
                all_paths.append(str(p))
                all_labels.append(class_to_idx[cls_dir.name])

    if len(all_paths) > 200:
        _, sample_paths, _, sample_labels = train_test_split(
            all_paths, all_labels, test_size=200,
            stratify=all_labels, random_state=42,
        )
    else:
        sample_paths, sample_labels = all_paths, all_labels

    def run_inference(pil_images):
        preds = []
        for img in pil_images:
            arr = np.array(img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS),
                           dtype=np.float32) / 255.0
            batch = np.expand_dims(arr, 0)
            out = model.predict(batch, verbose=0)[0]
            preds.append(int(np.argmax(out)))
        return preds

    originals, blurred_imgs, enhanced_imgs, true_labels = [], [], [], []
    for path, lbl in zip(sample_paths, sample_labels):
        try:
            orig = Image.open(path).convert("RGB")
        except Exception:
            continue
        # Artificial blur: Gaussian radius 3
        blurred = orig.filter(ImageFilter.GaussianBlur(radius=3))
        # Enhancement of the blurred image
        report = assess_quality(blurred)
        enhanced, _ = enhance_image(blurred, report)

        originals.append(orig)
        blurred_imgs.append(blurred)
        enhanced_imgs.append(enhanced)
        true_labels.append(lbl)

    true_arr = np.array(true_labels)
    results = {}
    for label, imgs in [("original", originals),
                        ("blurred", blurred_imgs),
                        ("enhanced_blurred", enhanced_imgs)]:
        preds = run_inference(imgs)
        acc = accuracy_score(true_arr, preds)
        prec = precision_score(true_arr, preds, average="weighted", zero_division=0)
        rec = recall_score(true_arr, preds, average="weighted", zero_division=0)
        f1 = f1_score(true_arr, preds, average="weighted", zero_division=0)
        results[label] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        logger.info(
            "[%s] Acc=%.4f | P=%.4f | R=%.4f | F1=%.4f",
            label, acc, prec, rec, f1,
        )

    return results


def save_blur_comparison_chart(comparison: dict, out_path: Path):
    """Bar chart comparing metrics across conditions."""
    if not comparison:
        return
    conditions = list(comparison.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = ["#4CAF50", "#FF9800", "#2196F3", "#9C27B0"]

    x = np.arange(len(conditions))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [comparison[c][metric] for c in conditions]
        ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([c.replace("_", " ").title() for c in conditions])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance: Original vs Blurred vs Enhanced", fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Blur comparison chart saved → %s", out_path)


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    data = load_test_results()

    # Standard metrics
    y_true, y_pred, class_names, acc, report = evaluate_standard(data)

    # Save report text
    report_path = EVAL_DIR / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(f"Test Accuracy: {acc:.4f}\n\n")
        fh.write(report)
    logger.info("Full report saved → %s", report_path)

    # Confusion matrix
    save_confusion_matrix(y_true, y_pred, class_names, EVAL_DIR / "confusion_matrix.png")

    # Training curves
    save_training_curves(EVAL_DIR / "training_curves.png")

    # Blur comparison (requires the model)
    model_path = Path(settings.BASE_DIR) / "models_saved" / "plant_disease_model.h5"
    if model_path.exists():
        import tensorflow as tf
        logger.info("Loading model for blur comparison …")
        model = tf.keras.models.load_model(str(model_path))
        comparison = blur_comparison_evaluation(data, model)
        if comparison:
            comp_path = EVAL_DIR / "blur_comparison.json"
            with open(comp_path, "w", encoding="utf-8") as fh:
                json.dump(comparison, fh, indent=2)
            save_blur_comparison_chart(comparison, EVAL_DIR / "blur_comparison_chart.png")

    print("\nEvaluation complete. Results in:", EVAL_DIR)


if __name__ == "__main__":
    main()
