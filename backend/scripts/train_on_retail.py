"""Build the retail-products-classification dataset and train YOLOv8 on it.

Runs anywhere the competition data is reachable:
  - Kaggle Notebooks, with the competition added as a data source (mounted at /kaggle/input)
  - Google Colab, after the download cell in notebooks/colab_train_retail.ipynb has unpacked
    it into /content/retail_data
  - locally, with --csv and --images-dir pointing at your own copy
"""

import argparse
import subprocess
import sys
from pathlib import Path

from prepare_retail_dataset import DATASET_DIRS

SCRIPTS_DIR = Path(__file__).resolve().parent


def _run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}")
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task",
        choices=sorted(DATASET_DIRS),
        default="classify",
        help="classify: train yolov8n-cls on this dataset's image-level labels (default). "
        "detect: train a detector on the full-frame-box shim",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild the dataset even if it already exists")
    parser.add_argument("--csv", default=None, help="Path to train.csv (or train.csv.zip); auto-detected if omitted")
    parser.add_argument("--images-dir", default=None, help="Directory with <ImgId>.<ext> product images")
    parser.add_argument("--weights", default=None, help="Default: yolov8n-cls.pt for classify, yolov8n.pt for detect")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=None, help="Default: 224 for classify, 640 for detect")
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
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap images kept per category, so one over-represented class can't dominate",
    )
    args = parser.parse_args()

    # classes.json is the last file prepare writes, so its presence means the build finished.
    built_marker = DATASET_DIRS[args.task] / "classes.json"
    if built_marker.exists() and not args.force:
        print(f"[skip] dataset already built at {built_marker.parent} (--force to rebuild)")
    else:
        prepare_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "prepare_retail_dataset.py"),
            "--layout",
            args.task,
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
        if args.max_per_class is not None:
            prepare_cmd += ["--max-per-class", str(args.max_per_class)]
        _run(prepare_cmd)

    train_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_yolo_retail.py"),
        "--task",
        args.task,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--cache",
        args.cache,
    ]
    if args.weights is not None:
        train_cmd += ["--weights", args.weights]
    if args.imgsz is not None:
        train_cmd += ["--imgsz", str(args.imgsz)]
    if args.device is not None:
        train_cmd += ["--device", args.device]
    _run(train_cmd)


if __name__ == "__main__":
    main()
