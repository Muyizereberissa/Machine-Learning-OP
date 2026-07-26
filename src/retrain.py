"""Retraining pipeline triggered by user-uploaded images (SQLite + disk)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    CLASS_NAMES,
    IMG_SIZE,
    MODEL_PATH,
    RETRAIN_EPOCHS,
    RETRAIN_MAX_IMAGES,
    SEED,
    UPLOAD_DIR,
)
from src.database import create_retrain_job, finish_retrain_job
from src.model import freeze_backbone_for_retrain, get_shared_model, load_model, save_model
from src.preprocessing import load_image_path


def _gather_upload_dataset(max_images: int = RETRAIN_MAX_IMAGES):
    xs, ys = [], []
    for label_idx, cls in enumerate(CLASS_NAMES):
        class_dir = UPLOAD_DIR / cls
        if not class_dir.exists():
            continue
        files = sorted(
            [
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        )
        for path in files:
            xs.append(load_image_path(path))
            ys.append(label_idx)
            if len(xs) >= max_images:
                break
        if len(xs) >= max_images:
            break

    if not xs:
        raise ValueError(
            "No uploaded images found. Upload labeled images under both classes first."
        )
    if len(set(ys)) < 2:
        raise ValueError(
            "Need uploaded images for BOTH Parasitized and Uninfected before retraining."
        )

    x = np.stack(xs).astype("float32")
    y = np.array(ys, dtype="float32")
    return x, y


def run_retrain(
    epochs: int = RETRAIN_EPOCHS,
    max_images: int = RETRAIN_MAX_IMAGES,
) -> dict:
    """
    Retrain using the saved custom model as a pretrained backbone.

    Pipeline:
    1. Load uploaded images saved to disk and SQLite
    2. Preprocess images
    3. Freeze convolutional layers and fine-tune the classification head
    """
    job_id = create_retrain_job()
    try:
        x, y = _gather_upload_dataset(max_images=max_images)
        n = len(x)

        # Shuffle and tiny holdout for metrics
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(n)
        x, y = x[idx], y[idx]
        split = max(1, int(0.2 * n))
        x_val, y_val = x[:split], y[:split]
        x_train, y_train = x[split:], y[split:]
        if len(x_train) < 2:
            x_train, y_train = x, y
            x_val, y_val = x, y

        model = load_model(MODEL_PATH)
        model = freeze_backbone_for_retrain(model)

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=min(16, len(x_train)),
            verbose=0,
        )

        probs = model.predict(x_val, verbose=0).ravel()
        preds = (probs >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_val, preds)),
            "precision": float(precision_score(y_val, preds, zero_division=0)),
            "recall": float(recall_score(y_val, preds, zero_division=0)),
            "f1": float(f1_score(y_val, preds, zero_division=0)),
            "loss": float(history.history["loss"][-1]),
            "val_loss": float(history.history.get("val_loss", [0])[-1]),
            "images_used": n,
            "epochs": epochs,
        }

        save_model(model, MODEL_PATH)
        get_shared_model(force_reload=True)

        finish_retrain_job(
            job_id,
            status="success",
            images_used=n,
            epochs=epochs,
            metrics_json=json.dumps(metrics),
            message="Retrain completed using uploaded images.",
        )
        return {"job_id": job_id, "status": "success", "metrics": metrics}
    except Exception as exc:
        finish_retrain_job(
            job_id,
            status="failed",
            images_used=0,
            epochs=0,
            metrics_json="{}",
            message=str(exc),
        )
        return {"job_id": job_id, "status": "failed", "error": str(exc)}


def make_tf_dataset_from_directory(directory: Path, shuffle: bool = True):
    """Build a TensorFlow image dataset from a class-folder directory."""
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=32,
        shuffle=shuffle,
        seed=SEED,
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
