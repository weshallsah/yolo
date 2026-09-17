"""End-to-end pipeline: extract the downloaded ABO archives, build the YOLO dataset, and train.

Wraps prepare_abo_dataset.py and train_yolo_abo.py so a single command takes you from the
raw ABO tarballs (already downloaded into training_data/abo/raw/) to a trained checkpoint.
Safe to re-run: extraction and dataset prep are skipped if their output already exists,
unless --force is passed.

Usage:
    python scripts/train_on_abo.py [--force] [--weights yolo11n.pt] [--epochs 30] [--imgsz 640] [--batch 16]
"""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
ABO_DIR = BACKEND_DIR / "training_data" / "abo"
RAW_DIR = ABO_DIR / "raw"
LISTINGS_TAR = RAW_DIR / "abo-listings.tar"
IMAGES_TAR = RAW_DIR / "abo-images-small.tar"
LISTINGS_DIR = ABO_DIR / "listings"
IMAGES_DIR = ABO_DIR / "images"
DATA_YAML_PATH = ABO_DIR / "dataset" / "data.yaml"
SCRIPTS_DIR = Path(__file__).resolve().parent


def _extract(tar_path: Path, dest_dir: Path, label: str, force: bool) -> None:
    if dest_dir.exists() and not force:
        print(f"[skip] {label} already extracted at {dest_dir}")
        return
    if not tar_path.exists():
        raise SystemExit(f"Missing {tar_path}. Download it into {RAW_DIR} first.")
    print(f"[extract] {tar_path.name} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest_dir)


def _run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}")
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-extract and rebuild the dataset even if present")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. 0 for first GPU, cpu for CPU (default: auto)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers; use 0 on Kaggle to avoid shm deadlocks")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--min-images-per-class", type=int, default=2)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap the total number of images used (randomly sampled across classes) before splitting",
    )
    args = parser.parse_args()

    _extract(LISTINGS_TAR, LISTINGS_DIR, "abo-listings", args.force)
    _extract(IMAGES_TAR, IMAGES_DIR, "abo-images-small", args.force)

    if DATA_YAML_PATH.exists() and not args.force:
        print(f"[skip] dataset already built at {DATA_YAML_PATH}")
    else:
        prepare_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "prepare_abo_dataset.py"),
            "--val-fraction",
            str(args.val_fraction),
            "--min-images-per-class",
            str(args.min_images_per_class),
        ]
        if args.max_images is not None:
            prepare_cmd += ["--max-images", str(args.max_images)]
        _run(prepare_cmd)

    train_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_yolo_abo.py"),
        "--weights",
        args.weights,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
    ]
    if args.device is not None:
        train_cmd += ["--device", args.device]
    _run(train_cmd)


if __name__ == "__main__":
    main()
