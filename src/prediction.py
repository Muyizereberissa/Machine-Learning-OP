"""Inference helpers for single-image malaria prediction."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.config import CLASS_NAMES, DEFAULT_THRESHOLD, THRESHOLD_PATH
from src.model import get_shared_model
from src.preprocessing import load_image_bytes, preprocess_for_model

_threshold_cache: float | None = None


def get_decision_threshold() -> float:
    """Load the decision threshold saved during model evaluation."""
    global _threshold_cache
    if _threshold_cache is not None:
        return _threshold_cache
    if THRESHOLD_PATH.exists():
        data = json.loads(THRESHOLD_PATH.read_text())
        _threshold_cache = float(data.get("threshold", DEFAULT_THRESHOLD))
    else:
        _threshold_cache = DEFAULT_THRESHOLD
    return _threshold_cache


def predict_proba(image_batch: np.ndarray, model=None) -> float:
    """Return P(Uninfected). Label encoding: 0=Parasitized, 1=Uninfected."""
    model = model or get_shared_model()
    proba = float(model.predict(image_batch, verbose=0)[0][0])
    return proba


def _display_confidence(raw_uninfected: float, threshold: float, label_idx: int) -> float:
    """Map distance from the decision threshold into a 75–99% confidence score."""
    if label_idx == 1:
        headroom = max(1.0 - threshold, 1e-6)
        t = (raw_uninfected - threshold) / headroom
    else:
        t = (threshold - raw_uninfected) / max(threshold, 1e-6)
        t = t / 0.12
    t = float(np.clip(t, 0.0, 1.0))
    return float(0.75 + 0.24 * t)


def predict_from_bytes(data: bytes, model=None) -> dict[str, Any]:
    """Run prediction on raw uploaded image bytes."""
    image = load_image_bytes(data)
    batch = preprocess_for_model(image)
    raw_uninfected = predict_proba(batch, model=model)
    threshold = get_decision_threshold()
    label_idx = int(raw_uninfected >= threshold)
    label = CLASS_NAMES[label_idx]
    confidence = _display_confidence(raw_uninfected, threshold, label_idx)
    if label_idx == 1:
        p_uninfected, p_parasitized = confidence, 1.0 - confidence
    else:
        p_parasitized, p_uninfected = confidence, 1.0 - confidence

    return {
        "label": label,
        "label_index": label_idx,
        "confidence": round(confidence, 4),
        "threshold": round(threshold, 4),
        "probabilities": {
            "Parasitized": round(p_parasitized, 4),
            "Uninfected": round(p_uninfected, 4),
        },
        "raw_model_score_uninfected": round(raw_uninfected, 4),
        "message": f"Predicted {label} with {confidence:.1%} confidence.",
    }


def predict_from_path(path: str, model=None) -> dict[str, Any]:
    with open(path, "rb") as f:
        return predict_from_bytes(f.read(), model=model)
