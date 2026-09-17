"""Build the retail-products-classification dataset (in YOLO format) and train on it.

Meant to run inside a Kaggle Notebook with the 'retail-products-classification' competition
added as a data source (mounted at /kaggle/input/retail-products-classification). No download
step is needed since Kaggle provides the data directly.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from prepare_retail_dataset import DATASET_DIR

SCRIPTS_DIR = Path(__file__).resolve().parent


def _run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}")
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Rebuild the dataset even if it already exists")
    parser.add_argument("--csv", default=None, help="Path to train.csv (or train.csv.zip); auto-detected if omitted")
    parser.add_argument("--images-dir", default=None, help="Directory with <ImgId>.<ext> product images")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=float,
        default=16,
        help="Fixed batch size (integer >= 1), a fraction between 0 and 1 to set AutoBatch's "
        "target GPU-memory utilization (e.g. 0.85 for ~85%%), or -1 for AutoBatch's default ~60%% target",
    )
    parser.add_argument("--device", default=None, help="e.g. 0 for first GPU, cpu for CPU (default: auto)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument(
        "--cache",
        choices=["ram", "disk", "none"],
        default="ram",
        help="Image caching strategy: ram (fastest, falls back to none if there isn't enough RAM), "
        "disk (caches resized images as .npy files, needs less RAM but more disk space), "
        "or none (reads from disk every batch)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--min-images-per-class", type=int, default=2)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap the total number of images used (randomly sampled across categories) before splitting",
    )
    args = parser.parse_args()

    data_yaml = DATASET_DIR / "data.yaml"
    if data_yaml.exists() and not args.force:
        print(f"[skip] dataset already built at {data_yaml}")
    else:
        prepare_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "prepare_retail_dataset.py"),
            "--val-fraction",
            str(args.val_fraction),
            "--min-images-per-class",
            str(args.min_images_per_class),
        ]
        if args.csv is not None:
            prepare_cmd += ["--csv", args.csv]
        if args.images_dir is not None:
            prepare_cmd += ["--images-dir", args.images_dir]
        if args.max_images is not None:
            prepare_cmd += ["--max-images", str(args.max_images)]
        _run(prepare_cmd)

    train_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_yolo_retail.py"),
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
        "--cache",
        args.cache,
    ]
    if args.device is not None:
        train_cmd += ["--device", args.device]
    _run(train_cmd)


if __name__ == "__main__":
    main()
