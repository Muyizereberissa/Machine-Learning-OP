"""Compact CNN for malaria cell classification."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMG_SIZE, LEARNING_RATE, MODEL_PATH


def configure_tf_memory() -> None:
    """Enable GPU memory growth when a GPU is available."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    tf.keras.backend.clear_session()


def build_malaria_cnn(
    input_shape: tuple[int, int, int] = (*IMG_SIZE, 3),
    dropout_rate: float = 0.4,
) -> keras.Model:
    """
    Lightweight CNN with Dropout, BatchNormalization, and Adam optimization
    for binary malaria cell classification.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(dropout_rate / 2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="malaria_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def get_callbacks(checkpoint_path: Path | None = None) -> list:
    """Early stopping + LR schedule + optional checkpointing."""
    checkpoint_path = checkpoint_path or MODEL_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]


def save_model(model: keras.Model, path: Path | None = None) -> Path:
    path = path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def load_model(path: Path | None = None) -> keras.Model:
    path = path or MODEL_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run scripts/train_initial.py first."
        )
    return keras.models.load_model(path)


def freeze_backbone_for_retrain(model: keras.Model) -> keras.Model:
    """Freeze convolutional layers and train only the dense classification head."""
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.BatchNormalization)):
            layer.trainable = False
        elif isinstance(layer, layers.Dense):
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 2),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


_model_singleton: Optional[keras.Model] = None


def get_shared_model(force_reload: bool = False) -> keras.Model:
    """Process-wide model singleton to avoid reloading on every request."""
    global _model_singleton
    if _model_singleton is None or force_reload:
        configure_tf_memory()
        _model_singleton = load_model()
    return _model_singleton
