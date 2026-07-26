#!/usr/bin/env python3
"""Train the compact malaria CNN and save models/malaria_cnn.keras."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODEL_PATH, TEST_DIR, THRESHOLD_PATH, TRAIN_DIR  # noqa: E402
from src.model import build_malaria_cnn, configure_tf_memory, get_callbacks, save_model  # noqa: E402
from src.retrain import make_tf_dataset_from_directory  # noqa: E402


def _collect(model, test_ds):
    y_true, y_prob = [], []
    for batch_x, batch_y in test_ds:
        probs = model.predict(batch_x, verbose=0).ravel()
        y_prob.extend(probs.tolist())
        y_true.extend(batch_y.numpy().ravel().tolist())
    return np.array(y_true), np.array(y_prob)


def calibrate_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    best = float(thr[np.argmax(tpr - fpr)])
    # roc_curve can return inf for the first threshold
    if not np.isfinite(best):
        best = 0.5
    THRESHOLD_PATH.write_text(json.dumps({"threshold": best}, indent=2))
    return best


def evaluate(model, test_ds) -> dict:
    y_true, y_prob = _collect(model, test_ds)
    threshold = calibrate_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "threshold": threshold,
        "loss": float(np.mean(-(y_true * np.log(y_prob + 1e-7) + (1 - y_true) * np.log(1 - y_prob + 1e-7)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    if not any(TRAIN_DIR.rglob("*.png")) and not any(TRAIN_DIR.rglob("*.jpg")):
        raise SystemExit("No training images found. Run scripts/download_data.py first.")

    import tensorflow as tf

    configure_tf_memory()
    train_ds = make_tf_dataset_from_directory(TRAIN_DIR, shuffle=True)
    test_ds = make_tf_dataset_from_directory(TEST_DIR, shuffle=False)

    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
        ],
        name="train_aug",
    )
    train_ds = train_ds.map(
        lambda x, y: (aug(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    model = build_malaria_cnn()
    model.summary()
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        callbacks=get_callbacks(MODEL_PATH),
    )
    # Reload best checkpoint weights if present
    if MODEL_PATH.exists():
        from src.model import load_model

        model = load_model(MODEL_PATH)
    metrics = evaluate(model, test_ds)
    save_model(model, MODEL_PATH)

    out = ROOT / "models" / "metrics.json"
    payload = {"metrics": metrics, "history": {k: [float(x) for x in v] for k, v in history.history.items()}}
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
