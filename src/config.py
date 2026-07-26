"""Shared paths and settings for the Malaria MLOps pipeline."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "app_data.db"))

MODEL_PATH = MODELS_DIR / "malaria_cnn.keras"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Parasitized", "Uninfected"]
BATCH_SIZE = 32
SEED = 42
DEFAULT_THRESHOLD = float(os.getenv("PREDICT_THRESHOLD", "0.5"))

RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "3"))
RETRAIN_MAX_IMAGES = int(os.getenv("RETRAIN_MAX_IMAGES", "200"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))

NIH_DATASET_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"

for path in (TRAIN_DIR, TEST_DIR, UPLOAD_DIR, RAW_DIR, MODELS_DIR):
    path.mkdir(parents=True, exist_ok=True)
