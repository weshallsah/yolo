import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

BACKEND_DIR = Path(__file__).resolve().parent.parent
ABO_DIR = BACKEND_DIR / "training_data" / "abo"
RAW_DIR = ABO_DIR / "raw"
LISTINGS_TAR = RAW_DIR / "abo-listings.tar"
IMAGES_TAR = RAW_DIR / "abo-images-small.tar"
LISTINGS_URL = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/archives/abo-listings.tar"
IMAGES_URL = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/archives/abo-images-small.tar"
LISTINGS_DIR = ABO_DIR / "listings"
IMAGES_DIR = ABO_DIR / "images"
DATA_YAML_PATH = ABO_DIR / "dataset_electronics" / "data.yaml"
SCRIPTS_DIR = Path(__file__).resolve().parent


def _download(url: str, dest_path: Path, force: bool) -> None:
    if dest_path.exists() and not force:
        print(f"[skip] {dest_path.name} already downloaded at {dest_path}")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest_path}")
    with urlopen(url) as response, open(dest_path, "wb") as out_file:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8 * 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1e6:,.0f} MB / {total / 1e6:,.0f} MB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r  {downloaded / 1e6:,.0f} MB", end="", flush=True)
    print()


def _extract(tar_path: Path, dest_dir: Path, label: str, force: bool) -> None:
    if dest_dir.exists() and not force:
        print(f"[skip] {label} already extracted at {dest_dir}")
        return
    if not tar_path.exists():
        raise SystemExit(f"Missing {tar_path}. Download it into {RAW_DIR} first.")
    print(f"[extract] {tar_path.name} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("tar"):
        subprocess.run(["tar", "-xf", str(tar_path), "-C", str(dest_dir)], check=True)
    else:
        with tarfile.open(tar_path) as tar:
            tar.extractall(dest_dir)


def _run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}")
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-extract and rebuild the dataset even if present")
    parser.add_argument("--weights", default="yolov8n.pt")
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

    _download(LISTINGS_URL, LISTINGS_TAR, args.force)
    _download(IMAGES_URL, IMAGES_TAR, args.force)
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
