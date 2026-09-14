"""Fine-tunes the base YOLO checkpoint on new object classes it couldn't detect.

The app's identify pipeline falls back to Google Lens whenever YOLO can't confidently
name an object, and (see app/training/dataset_recorder.py) saves that image into
training_data/pool/<class-slug>/ under the generic name Lens gave it (e.g. "helmet").
This script turns that accumulated pool into a real YOLO dataset and fine-tunes from
the current checkpoint:

  - Every pooled image gets a full-frame bounding box, since this app's photos are one
    product filling most of the frame.
  - New classes are assigned stable ids after the base checkpoint's existing classes
    (persisted in training_data/custom_classes.json across runs).
  - Ultralytics' small COCO128 sample is mixed in so fine-tuning doesn't erase the
    base checkpoint's original classes (we don't have the full COCO dataset locally).

Usage:
    python scripts/train_yolo.py [--weights yolo11n.pt] [--epochs 30] [--imgsz 640]

After training, validate the printed metrics, then point APP_YOLO_WEIGHTS_PATH at
models/custom_yolo/weights/best.pt to start serving the new classes.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

BACKEND_DIR = Path(__file__).resolve().parent.parent
POOL_DIR = BACKEND_DIR / "training_data" / "pool"
CLASS_MAP_PATH = BACKEND_DIR / "training_data" / "custom_classes.json"
DATASET_DIR = BACKEND_DIR / "training_data" / "dataset"
MODELS_DIR = BACKEND_DIR / "models"
VAL_FRACTION = 0.15
FULL_FRAME_BOX = "0.5 0.5 0.98 0.98"  # class_id cx cy w h, normalized, small margin off the edges


def _load_base_names(weights_path: str) -> dict[int, str]:
    return YOLO(weights_path).names


def _load_or_assign_custom_class_ids(base_names: dict[int, str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    if CLASS_MAP_PATH.exists():
        mapping = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))

    next_id = max((*base_names, *mapping.values(), -1)) + 1
    for class_dir in sorted(p for p in POOL_DIR.iterdir() if p.is_dir()):
        if class_dir.name not in mapping:
            mapping[class_dir.name] = next_id
            next_id += 1

    CLASS_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_MAP_PATH.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def _reset_dataset_dir() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def _add_custom_classes(class_ids: dict[str, int], seed: int) -> int:
    rng = random.Random(seed)
    count = 0
    for class_dir in POOL_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        class_id = class_ids[class_dir.name]
        images = sorted(class_dir.glob("*.jpg"))
        rng.shuffle(images)
        split_at = len(images) if len(images) <= 1 else max(1, int(len(images) * (1 - VAL_FRACTION)))

        for index, image_path in enumerate(images):
            split = "train" if index < split_at else "val"
            dest_stem = f"{class_dir.name}_{image_path.stem}"
            shutil.copy(image_path, DATASET_DIR / "images" / split / f"{dest_stem}.jpg")
            (DATASET_DIR / "labels" / split / f"{dest_stem}.txt").write_text(
                f"{class_id} {FULL_FRAME_BOX}\n", encoding="utf-8"
            )
            count += 1
    return count


def _add_coco128_sample() -> int:
    """Mixes in Ultralytics' small COCO128 sample (downloaded on first use) so
    fine-tuning doesn't erase the base checkpoint's original classes."""
    info = check_det_dataset("coco128.yaml")
    count = 0

    for split in ("train", "val"):
        images_dir = Path(info[split])
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        if not images_dir.exists() or not labels_dir.exists():
            continue

        for image_path in images_dir.glob("*.jpg"):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            shutil.copy(image_path, DATASET_DIR / "images" / split / f"coco128_{image_path.name}")
            shutil.copy(label_path, DATASET_DIR / "labels" / split / f"coco128_{image_path.stem}.txt")
            count += 1

    return count


def _write_data_yaml(base_names: dict[int, str], class_ids: dict[str, int]) -> Path:
    names = {**base_names, **{class_id: name for name, class_id in class_ids.items()}}
    data = {
        "path": str(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {i: names[i] for i in sorted(names)},
    }
    data_yaml_path = DATASET_DIR / "data.yaml"
    data_yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yolo11n.pt", help="Base checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-coco128", action="store_true", help="Skip mixing in the COCO128 sample (faster, more forgetting risk)"
    )
    args = parser.parse_args()

    if not POOL_DIR.exists() or not any(p.is_dir() for p in POOL_DIR.iterdir()):
        raise SystemExit(
            f"No collected samples found in {POOL_DIR}. Nothing to train on yet — "
            "scan objects that YOLO can't detect (they fall back to Google Lens) first."
        )

    base_names = _load_base_names(args.weights)
    class_ids = _load_or_assign_custom_class_ids(base_names)

    _reset_dataset_dir()
    custom_count = _add_custom_classes(class_ids, args.seed)
    coco_count = 0 if args.skip_coco128 else _add_coco128_sample()
    data_yaml_path = _write_data_yaml(base_names, class_ids)

    print(f"New classes: {class_ids}")
    print(f"Training on {custom_count} new-class samples + {coco_count} COCO128 samples.\n")

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(MODELS_DIR),
        name="custom_yolo",
        exist_ok=True,
    )

    print(
        "\nDone. Check the validation metrics above (especially per-class mAP for the "
        f"new classes {list(class_ids)}), then point APP_YOLO_WEIGHTS_PATH at "
        f"{MODELS_DIR / 'custom_yolo' / 'weights' / 'best.pt'} to start serving it."
    )


if __name__ == "__main__":
    main()
