"""Builds a YOLO-format dataset from the Amazon Berkeley Objects (ABO) catalog.

ABO ships product photos and rich metadata (see abo-listings.tar) but no bounding-box
annotations — it's a retrieval/classification dataset, not a detection one. Like
app/training/dataset_recorder.py does for the app's own self-collected samples, we treat
each catalog photo as one object filling the frame (ABO product shots are studio photos
of a single item) and assign it a full-frame box labeled by its `product_type`
(e.g. "SHOES", "CHAIR").

Expects these already downloaded and extracted under training_data/abo/:
  - raw/abo-listings.tar     -> extracted to listings/   (per-item metadata, gzipped JSONL)
  - raw/abo-images-small.tar -> extracted to images/      (downscaled photos + images.csv.gz)

Only keeps listings whose product_type is in ELECTRONICS_PRODUCT_TYPES below, so the
resulting dataset/model is scoped to electronics instead of ABO's full ~505-category catalog.

Usage:
    python scripts/prepare_abo_dataset.py [--val-fraction 0.15] [--min-images-per-class 2]
"""

import argparse
import csv
import gzip
import json
import random
import shutil
from pathlib import Path

import yaml

BACKEND_DIR = Path(__file__).resolve().parent.parent
ABO_DIR = BACKEND_DIR / "training_data" / "abo"
LISTINGS_DIR = ABO_DIR / "listings" / "listings" / "metadata"
IMAGES_DIR = ABO_DIR / "images"
IMAGES_METADATA_CSV = IMAGES_DIR / "images" / "metadata" / "images.csv.gz"
IMAGES_ROOT = IMAGES_DIR / "images" / "small"
DATASET_DIR = ABO_DIR / "dataset_electronics"
FULL_FRAME_BOX = "0.5 0.5 0.98 0.98"  # class_id cx cy w h, normalized, small margin off the edges

ELECTRONICS_PRODUCT_TYPES = {
    "ABIS_ELECTRONICS", "AMAZON_TABLET_ACCESSORY", "ANTENNA", "AUDIO_OR_VIDEO", "BATTERY",
    "BINOCULAR", "CALCULATOR", "CAMCORDER", "CAMERA_BAGS_AND_CASES", "CAMERA_FLASH",
    "CAMERA_LENS_FILTERS", "CAMERA_OTHER_ACCESSORIES", "CAMERA_SUPPORT", "CAMERA_TRIPOD",
    "CAR_ELECTRONICS", "CE_ACCESSORY", "CE_CARRYING_CASE_OR_BAG", "CELLULAR_PHONE_CASE",
    "CHARGING_ADAPTER", "COMPUTER", "COMPUTER_ADD_ON", "COMPUTER_COMPONENT",
    "COMPUTER_DRIVE_OR_STORAGE", "COMPUTER_INPUT_DEVICE", "COMPUTER_INPUT_DEVICE_ACCESSORY",
    "COMPUTER_SPEAKER", "CONSUMER_ELECTRONICS", "ELECTRONIC_ADAPTER", "ELECTRONIC_CABLE",
    "ELECTRONIC_DEVICE_DOCKING_STATION", "ELECTRONIC_SWITCH", "FLASH_DRIVE", "FLASH_MEMORY",
    "GPS_OR_NAVIGATION_ACCESSORY", "GPS_OR_NAVIGATION_SYSTEM", "HEADPHONES", "INPUT_MOUSE",
    "INPUT_PEN", "KEYBOARDS", "LANDLINE_PHONE", "MICROPHONE", "MICROSCOPES", "MULTIPORT_HUB",
    "NETWORKING_DEVICE", "NETWORK_INTERFACE_CONTROLLER_ADAPTER", "NOTEBOOK_COMPUTER",
    "OFFICE_ELECTRONICS", "PERSONAL_COMPUTER", "PHONE", "PHONE_ACCESSORY", "PORTABLE_AUDIO",
    "PORTABLE_AV_DEVICE", "PORTABLE_ELECTRONIC_DEVICE_COVER", "PORTABLE_ELECTRONIC_DEVICE_MOUNT",
    "PORTABLE_ELECTRONIC_DEVICE_STAND", "POWER_CONVERTER", "POWER_STRIP",
    "POWER_SUPPLIES_OR_PROTECTION", "PRINTER", "RADIO", "RECEIVER_OR_AMPLIFIER",
    "REMOTE_CONTROL", "SCREEN_PROTECTOR", "SECURITY_ELECTRONICS", "SPEAKERS",
    "SURVEILANCE_SYSTEMS", "SYSTEM_POWER_DEVICE", "TELEVISION", "TWO_WAY_RADIO",
    "VIDEO_GAME_ACCESSORIES", "VIDEO_PROJECTOR", "WIRELESS_ACCESSORY",
}


def _load_image_paths() -> dict[str, str]:
    """Maps image_id -> path relative to IMAGES_ROOT, from images/metadata/images.csv.gz."""
    if not IMAGES_METADATA_CSV.exists():
        raise SystemExit(
            f"Missing {IMAGES_METADATA_CSV}. Download and extract abo-images-small.tar into "
            f"{IMAGES_DIR} first."
        )
    paths: dict[str, str] = {}
    with gzip.open(IMAGES_METADATA_CSV, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paths[row["image_id"]] = row["path"]
    return paths


def _load_item_classes() -> dict[str, str]:
    """Maps main_image_id -> product_type for every listing that has both."""
    if not LISTINGS_DIR.exists():
        raise SystemExit(f"Missing {LISTINGS_DIR}. Download and extract abo-listings.tar first.")

    image_to_class: dict[str, str] = {}
    for shard in sorted(LISTINGS_DIR.glob("listings_*.json.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                image_id = record.get("main_image_id")
                product_types = record.get("product_type")
                if not image_id or not product_types:
                    continue
                product_type = product_types[0]["value"]
                if product_type not in ELECTRONICS_PRODUCT_TYPES:
                    continue
                image_to_class[image_id] = product_type
    return image_to_class


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
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--min-images-per-class",
        type=int,
        default=2,
        help="Classes with fewer images than this can't be split into train/val and are dropped",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap the total number of images used (randomly sampled across classes) before splitting",
    )
    args = parser.parse_args()

    print("Loading image path index...")
    image_paths = _load_image_paths()
    print(f"  {len(image_paths):,} images indexed")

    print("Loading item -> product_type labels...")
    image_classes = _load_item_classes()
    print(f"  {len(image_classes):,} listings with a main image and product type")

    by_class: dict[str, list[str]] = {}
    missing_files = 0
    for image_id, class_name in image_classes.items():
        rel_path = image_paths.get(image_id)
        if rel_path is None:
            missing_files += 1
            continue
        by_class.setdefault(class_name, []).append(rel_path)

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
    print(f"{len(dropped)} classes dropped for too few images ({sum(len(p) for p in dropped.values())} images)")
    if missing_files:
        print(f"{missing_files} listings skipped: image file not found in images.csv.gz")

    _reset_dataset_dir()
    class_ids = {name: i for i, name in enumerate(class_names)}
    rng = random.Random(args.seed)

    total_train = total_val = 0
    for class_name, rel_paths in by_class.items():
        class_id = class_ids[class_name]
        rng.shuffle(rel_paths)
        split_at = max(1, int(len(rel_paths) * (1 - args.val_fraction)))

        for index, rel_path in enumerate(rel_paths):
            split = "train" if index < split_at else "val"
            src = IMAGES_ROOT / rel_path
            if not src.exists():
                continue
            dest_stem = Path(rel_path).stem
            dest_name = f"{class_name.lower()}_{dest_stem}"
            shutil.copy(src, DATASET_DIR / "images" / split / f"{dest_name}.jpg")
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
