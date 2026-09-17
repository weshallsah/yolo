import argparse
import csv
import io
import json
import random
import re
import shutil
import zipfile
from pathlib import Path

import yaml

BACKEND_DIR = Path(__file__).resolve().parent.parent
KAGGLE_INPUT_DIR = Path("/kaggle/input/retail-products-classification")
DEFAULT_CSV_CANDIDATES = [
    KAGGLE_INPUT_DIR / "train.csv",
    BACKEND_DIR / "train.csv",
    BACKEND_DIR / "train.csv.zip",
]
DEFAULT_IMAGES_DIR = KAGGLE_INPUT_DIR / "train"
DATASET_DIR = BACKEND_DIR / "training_data" / "retail" / "dataset_products"
FULL_FRAME_BOX = "0.5 0.5 0.98 0.98"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _find_default_csv() -> Path:
    for candidate in DEFAULT_CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Could not find train.csv. Pass --csv explicitly, or (on Kaggle) add the "
        f"'retail-products-classification' competition as a data source so it's mounted at {KAGGLE_INPUT_DIR}"
    )


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if csv_path.suffix == ".zip":
        with zipfile.ZipFile(csv_path) as zf:
            names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not names:
                raise SystemExit(f"No .csv file found inside {csv_path}")
            with zf.open(names[0]) as raw:
                return list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")))
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _resolve_image(images_dir: Path, image_id: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{image_id}{extension}"
        if candidate.exists():
            return candidate
    matches = list(images_dir.glob(f"{image_id}.*"))
    return matches[0] if matches else None


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _reset_dataset_dir() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_data_yaml(class_names: list[str]) -> Path:
    data = {
        "path": str(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(class_names)),
    }
    data_yaml_path = DATASET_DIR / "data.yaml"
    data_yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=None, help="Path to train.csv (or train.csv.zip); auto-detected if omitted")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing the product images, named <ImgId>.<ext>",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--min-images-per-class",
        type=int,
        default=2,
        help="Categories with fewer images than this can't be split into train/val and are dropped",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap the total number of images used (randomly sampled across categories) before splitting",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = args.csv or _find_default_csv()
    print(f"Reading labels from {csv_path}")
    rows = _read_rows(csv_path)
    print(f"  {len(rows):,} rows")

    if not args.images_dir.exists():
        raise SystemExit(
            f"Missing images directory {args.images_dir}. On Kaggle, add the "
            "'retail-products-classification' competition as a data source to this notebook, "
            "or pass --images-dir to point at wherever the images actually live."
        )

    by_class: dict[str, list[str]] = {}
    missing_files = 0
    for row in rows:
        image_id = row["ImgId"]
        category = row["categories"]
        image_path = _resolve_image(args.images_dir, image_id)
        if image_path is None:
            missing_files += 1
            continue
        by_class.setdefault(category, []).append(str(image_path))

    if args.max_images is not None:
        all_pairs = [(name, path) for name, paths in by_class.items() for path in paths]
        random.Random(args.seed).shuffle(all_pairs)
        all_pairs = all_pairs[: args.max_images]
        by_class = {}
        for name, path in all_pairs:
            by_class.setdefault(name, []).append(path)
        print(f"Capped to {len(all_pairs):,} images (--max-images {args.max_images})")

    dropped = {name: paths for name, paths in by_class.items() if len(paths) < args.min_images_per_class}
    by_class = {name: paths for name, paths in by_class.items() if len(paths) >= args.min_images_per_class}
    class_names = sorted(by_class)

    print(f"\n{len(class_names)} classes usable (>= {args.min_images_per_class} images each)")
    if dropped:
        print(f"{len(dropped)} classes dropped for too few images ({sum(len(p) for p in dropped.values())} images)")
    if missing_files:
        print(f"{missing_files:,} rows skipped: image file not found in {args.images_dir}")

    _reset_dataset_dir()
    class_ids = {name: i for i, name in enumerate(class_names)}
    rng = random.Random(args.seed)

    total_train = total_val = 0
    for class_name, paths in by_class.items():
        class_id = class_ids[class_name]
        rng.shuffle(paths)
        split_at = max(1, int(len(paths) * (1 - args.val_fraction)))

        for index, src_path in enumerate(paths):
            split = "train" if index < split_at else "val"
            src = Path(src_path)
            dest_name = f"{_slugify(class_name)}_{src.stem}"
            shutil.copy(src, DATASET_DIR / "images" / split / f"{dest_name}{src.suffix}")
            (DATASET_DIR / "labels" / split / f"{dest_name}.txt").write_text(
                f"{class_id} {FULL_FRAME_BOX}\n", encoding="utf-8"
            )
            if split == "train":
                total_train += 1
            else:
                total_val += 1

    data_yaml_path = _write_data_yaml(class_names)
    (DATASET_DIR / "classes.json").write_text(json.dumps(class_ids, indent=2), encoding="utf-8")

    print(f"\nDataset ready: {total_train:,} train / {total_val:,} val images across {len(class_names)} classes")
    print(f"data.yaml written to {data_yaml_path}")


if __name__ == "__main__":
    main()
