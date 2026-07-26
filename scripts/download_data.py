#!/usr/bin/env python3
"""Download NIH malaria dataset and create train/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import create_splits, download_dataset, dataset_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1500,
        help="Cap images per class for faster local/CPU training (default: 1500).",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    source = download_dataset(force=args.force_download)
    create_splits(
        source_dir=source,
        max_per_class=args.max_per_class,
        rebuild=args.rebuild,
    )
    print("Train:", dataset_counts())
    from src.config import TEST_DIR
    from src.preprocessing import dataset_counts as counts

    print("Test:", counts(TEST_DIR))


if __name__ == "__main__":
    main()
