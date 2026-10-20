"""
detector/ml/train.py
====================
CNN training script for plant disease classification.

Usage (run from the plant_disease_project directory with the venv active):
    python detector/ml/train.py

Requirements
------------
* PlantVillage dataset must be placed at:
    plant_disease_project/dataset/PlantVillage/
  with one subdirectory per class, e.g.:
    dataset/PlantVillage/Apple___Apple_scab/
    dataset/PlantVillage/Apple___healthy/
    ...

Outputs (written to plant_disease_project/models_saved/)
---------
* plant_disease_model.h5   – trained Keras model
* class_labels.json        – {index: class_name} mapping
* training_history.json    – per-epoch loss/accuracy curves
"""

from __future__ import annotations

import json
import os
import sys
import logging
from pathlib import Path

# ── Add project root to PYTHONPATH so Django settings are importable ─────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from django.conf import settings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

import argparse
import random

# ── Configuration ────────────────────────────────────────────────────────────
def find_dataset_dir(explicit_path: str | None = None) -> Path:
    if explicit_path and Path(explicit_path).exists():
        return Path(explicit_path)
    candidates = [
        Path(settings.BASE_DIR) / "dataset" / "PlantVillage",
        Path(settings.BASE_DIR) / "dataset",
        Path(settings.BASE_DIR).parent / "dataset" / "PlantVillage",
        Path(settings.BASE_DIR).parent / "dataset",
    ]
    for c in candidates:
        if c.exists():
            subdirs = [d.name for d in c.iterdir() if d.is_dir()]
            if any("Tomato" in s or "Pepper" in s or "Potato" in s for s in subdirs):
                return c
    return Path(settings.BASE_DIR) / "dataset" / "PlantVillage"


MODEL_SAVE_DIR = Path(settings.BASE_DIR) / "models_saved"
MODEL_PATH = MODEL_SAVE_DIR / "plant_disease_model.h5"
LABELS_PATH = MODEL_SAVE_DIR / "class_labels.json"
HISTORY_PATH = MODEL_SAVE_DIR / "training_history.json"

IMG_SIZE = settings.CNN_INPUT_SIZE          # (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.10
SEED = 42


# ── Dataset discovery ─────────────────────────────────────────────────────────
def discover_dataset(dataset_dir: Path, max_per_class: int = 0):
    """
    Walk the PlantVillage directory structure and return parallel lists of
    (image_path, class_index) tuples plus the sorted class names.
    Ignores non-class directories (like nested 'PlantVillage').
    """
    if not dataset_dir.exists():
        logger.error(
            "Dataset directory not found: %s\n"
            "Please copy the PlantVillage dataset to that location so that "
            "each class is a sub-folder containing its images.",
            dataset_dir,
        )
        sys.exit(1)

    class_dirs = sorted([
        d for d in dataset_dir.iterdir()
        if d.is_dir() and d.name.lower() != "plantvillage"
    ])
    if not class_dirs:
        logger.error("No class sub-directories found in %s", dataset_dir)
        sys.exit(1)

    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    image_paths, labels = [], []
    for cls_dir in class_dirs:
        cls_imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
            cls_imgs.extend(list(cls_dir.glob(ext)))
        if max_per_class > 0 and len(cls_imgs) > max_per_class:
            rng = random.Random(SEED)
            cls_imgs = rng.sample(cls_imgs, max_per_class)
        for img_path in cls_imgs:
            image_paths.append(str(img_path))
            labels.append(class_to_idx[cls_dir.name])

    logger.info("Discovered %d classes, %d images.", len(class_names), len(image_paths))
    return image_paths, labels, class_names


# ── tf.data pipeline ─────────────────────────────────────────────────────────
def _load_and_preprocess(path: str, label: int, augment: bool = False):
    """TF graph function: read → decode → resize → normalise [→ augment]."""
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.ensure_shape(img, (*IMG_SIZE, 3))
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.clip_by_value(img, 0.0, 1.0)

    return img, label


def make_dataset(paths, labels, batch_size: int, augment: bool, shuffle: bool):
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(labels, dtype=tf.int32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, l: _load_and_preprocess(p, l, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Model definition ──────────────────────────────────────────────────────────
def build_model(num_classes: int) -> keras.Model:
    """
    Transfer-learning model based on MobileNetV2 (pretrained on ImageNet).
    The base is initially frozen; a second fine-tuning phase unfreezes the top.
    """
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # freeze for initial training

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="plant_disease_cnn")
    return model, base


# ── Training loop ─────────────────────────────────────────────────────────────
def train(
    dataset_dir: Path | None = None,
    phase1_epochs: int = 5,
    phase2_epochs: int = 3,
    batch_size: int = 32,
    max_per_class: int = 250,
):
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if dataset_dir is None:
        dataset_dir = find_dataset_dir()
    logger.info("Using dataset directory: %s", dataset_dir)

    # 1. Discover dataset
    image_paths, labels, class_names = discover_dataset(dataset_dir, max_per_class=max_per_class)
    num_classes = len(class_names)

    # 2. Stratified split: train / val / test (no leakage)
    paths_arr = np.array(image_paths)
    labels_arr = np.array(labels)

    # First, carve out the held-out test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths_arr, labels_arr,
        test_size=TEST_SPLIT,
        stratify=labels_arr,
        random_state=SEED,
    )
    # Then split train vs validation
    val_fraction_of_remaining = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_fraction_of_remaining,
        stratify=train_val_labels,
        random_state=SEED,
    )

    logger.info(
        "Split — train: %d | val: %d | test: %d",
        len(train_paths), len(val_paths), len(test_paths),
    )

    # 3. Save class-label mapping
    class_labels = {str(i): name for i, name in enumerate(class_names)}
    with open(LABELS_PATH, "w", encoding="utf-8") as fh:
        json.dump(class_labels, fh, indent=2)
    logger.info("Class labels saved → %s", LABELS_PATH)

    # 4. Build tf.data pipelines
    train_ds = make_dataset(train_paths.tolist(), train_labels.tolist(),
                            batch_size, augment=True, shuffle=True)
    val_ds   = make_dataset(val_paths.tolist(), val_labels.tolist(),
                            batch_size, augment=False, shuffle=False)
    test_ds  = make_dataset(test_paths.tolist(), test_labels.tolist(),
                            batch_size, augment=False, shuffle=False)

    # 5. Build model
    model, base = build_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    # 6. Phase 1: train head only
    logger.info("=== Phase 1: Training head (%d epochs, base frozen) ===", phase1_epochs)
    history1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=phase1_epochs, callbacks=callbacks,
    )

    # 7. Phase 2: fine-tune top layers of the base
    history2 = None
    if phase2_epochs > 0:
        logger.info("=== Phase 2: Fine-tuning top 40 layers of MobileNetV2 (%d epochs) ===", phase2_epochs)
        base.trainable = True
        for layer in base.layers[:-40]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=phase2_epochs, callbacks=callbacks,
        )

    # 8. Save final model (.h5 and .keras)
    model.save(str(MODEL_PATH))
    keras_path = MODEL_PATH.with_suffix(".keras")
    try:
        model.save(str(keras_path))
    except Exception as e:
        logger.warning("Could not save .keras format: %s", e)
    logger.info("Model saved → %s", MODEL_PATH)

    # 9. Persist training history
    def _h(h): return {k: [float(v) for v in vals] for k, vals in h.history.items()} if h else {}
    combined_history = {
        "phase1": _h(history1),
        "phase2": _h(history2),
    }
    with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
        json.dump(combined_history, fh, indent=2)
    logger.info("Training history saved → %s", HISTORY_PATH)

    # 10. Test-set evaluation
    logger.info("Evaluating on held-out test set …")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    logger.info("Test accuracy: %.4f | Test loss: %.4f", test_acc, test_loss)

    # 11. Persist test-set ground truth + predictions for evaluate.py
    all_true, all_pred = [], []
    for imgs, lbls in test_ds:
        preds = model.predict(imgs, verbose=0)
        all_pred.extend(np.argmax(preds, axis=1).tolist())
        all_true.extend(lbls.numpy().tolist())

    eval_data = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "class_names": class_names,
        "y_true": all_true,
        "y_pred": all_pred,
    }
    eval_path = Path(settings.BASE_DIR) / "evaluation" / "test_results.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as fh:
        json.dump(eval_data, fh, indent=2)
    logger.info("Test results saved → %s", eval_path)

    print("\n" + "=" * 60)
    print(f"  Training complete.")
    print(f"  Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Model saved   : {MODEL_PATH}")
    print("=" * 60)
    return test_acc, test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN model on PlantVillage dataset")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to PlantVillage directory")
    parser.add_argument("--phase1-epochs", type=int, default=5, help="Epochs for Phase 1 (frozen base)")
    parser.add_argument("--phase2-epochs", type=int, default=3, help="Epochs for Phase 2 (fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-per-class", type=int, default=250, help="Max images per class (0 for all)")
    args = parser.parse_args()

    train(
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
    )
