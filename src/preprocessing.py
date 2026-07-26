"""Data acquisition helpers and image preprocessing for malaria cell images."""

from __future__ import annotations

import io
import random
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Tuple
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

from src.config import (
    CLASS_NAMES,
    IMG_SIZE,
    NIH_DATASET_URL,
    RAW_DIR,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
)


def download_dataset(force: bool = False) -> Path:
    """Download and extract the NIH malaria cell_images dataset."""
    extract_dir = RAW_DIR / "cell_images"
    zip_path = RAW_DIR / "cell_images.zip"

    if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
        return extract_dir

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if force or not zip_path.exists():
        print(f"Downloading dataset from {NIH_DATASET_URL} ...")
        urlretrieve(NIH_DATASET_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    # Archive layout is usually raw/cell_images/{Parasitized,Uninfected}
    if not extract_dir.exists():
        nested = RAW_DIR / "cell_images"
        if nested.exists():
            return nested
    return extract_dir


def _list_class_images(source_dir: Path, class_name: str) -> list[Path]:
    class_dir = source_dir / class_name
    if not class_dir.exists():
        return []
    return sorted(
        [
            p
            for p in class_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )


def create_splits(
    source_dir: Path | None = None,
    train_ratio: float = 0.8,
    max_per_class: int | None = None,
    rebuild: bool = False,
) -> Tuple[Path, Path]:
    """Create train/test folders with an 80/20 split (seed=42)."""
    source_dir = source_dir or (RAW_DIR / "cell_images")
    marker = TRAIN_DIR / ".split_done"

    if marker.exists() and not rebuild and any(TRAIN_DIR.rglob("*.png")):
        return TRAIN_DIR, TEST_DIR

    if rebuild:
        for root in (TRAIN_DIR, TEST_DIR):
            if root.exists():
                shutil.rmtree(root)

    rng = random.Random(SEED)
    for cls in CLASS_NAMES:
        images = _list_class_images(source_dir, cls)
        if not images:
            raise FileNotFoundError(
                f"No images found for class '{cls}' in {source_dir / cls}"
            )
        rng.shuffle(images)
        if max_per_class is not None:
            images = images[:max_per_class]

        split_idx = int(len(images) * train_ratio)
        train_imgs, test_imgs = images[:split_idx], images[split_idx:]

        for subset, paths in ((TRAIN_DIR, train_imgs), (TEST_DIR, test_imgs)):
            dest = subset / cls
            dest.mkdir(parents=True, exist_ok=True)
            for src in paths:
                target = dest / src.name
                if not target.exists():
                    shutil.copy2(src, target)

    marker.write_text("ok")
    return TRAIN_DIR, TEST_DIR


def load_image_bytes(data: bytes, img_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Decode image bytes into a normalized float32 array shaped (H, W, 3)."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.resize(img_size, Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def load_image_path(path: Path | str, img_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Load an image file into a normalized float32 array."""
    with open(path, "rb") as f:
        return load_image_bytes(f.read(), img_size=img_size)


def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """Add batch dimension for model inference."""
    if image.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape {image.shape}")
    return np.expand_dims(image, axis=0)


def iter_labeled_paths(root: Path) -> Iterable[Tuple[Path, int]]:
    """Yield (path, label_index) for Parasitized=0, Uninfected=1."""
    for label, cls in enumerate(CLASS_NAMES):
        class_dir = root / cls
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                yield path, label


def dataset_counts(root: Path | None = None) -> dict:
    """Return per-class image counts for insights."""
    root = root or TRAIN_DIR
    counts = {}
    for cls in CLASS_NAMES:
        class_dir = root / cls
        n = (
            len(
                [
                    p
                    for p in class_dir.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ]
            )
            if class_dir.exists()
            else 0
        )
        counts[cls] = n
    counts["total"] = sum(counts[c] for c in CLASS_NAMES)
    return counts


def summarize_image_stats(root: Path | None = None, sample_size: int = 200) -> dict:
    """Sample images and compute mean brightness, red, and green channel stats."""
    root = root or TRAIN_DIR
    rng = random.Random(SEED)
    paths = [p for p, _ in iter_labeled_paths(root)]
    if not paths:
        return {
            "sample_size": 0,
            "mean_brightness": 0.0,
            "mean_red": 0.0,
            "mean_green": 0.0,
            "by_class": {},
        }

    sample = paths if len(paths) <= sample_size else rng.sample(paths, sample_size)
    brightness, reds, greens = [], [], []
    by_class = {cls: {"brightness": [], "red": [], "green": []} for cls in CLASS_NAMES}

    for path in sample:
        arr = load_image_path(path)
        b = float(arr.mean())
        r = float(arr[:, :, 0].mean())
        g = float(arr[:, :, 1].mean())
        brightness.append(b)
        reds.append(r)
        greens.append(g)
        cls = path.parent.name
        if cls in by_class:
            by_class[cls]["brightness"].append(b)
            by_class[cls]["red"].append(r)
            by_class[cls]["green"].append(g)

    def _avg(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "sample_size": len(sample),
        "mean_brightness": _avg(brightness),
        "mean_red": _avg(reds),
        "mean_green": _avg(greens),
        "by_class": {
            cls: {
                "mean_brightness": _avg(stats["brightness"]),
                "mean_red": _avg(stats["red"]),
                "mean_green": _avg(stats["green"]),
                "n": len(stats["brightness"]),
            }
            for cls, stats in by_class.items()
        },
    }


def save_upload_image(data: bytes, dest_dir: Path, filename: str, label: str) -> Path:
    """Validate and save an uploaded image under data/uploads/<label>/."""
    if label not in CLASS_NAMES:
        raise ValueError(f"Label must be one of {CLASS_NAMES}")
    # Ensure bytes decode as an image
    load_image_bytes(data)
    target_dir = dest_dir / label
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    target = target_dir / safe_name
    target.write_bytes(data)
    return target
