import argparse
import collections
import concurrent.futures
import csv
import hashlib
import io
import json
import random
import re
import shutil
import zipfile
from pathlib import Path

import yaml
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent
KAGGLE_INPUT_DIR = Path("/kaggle/input")
DEFAULT_CSV_CANDIDATES = [
    KAGGLE_INPUT_DIR / "retail-products-classification" / "train.csv",
    BACKEND_DIR / "train.csv",
    BACKEND_DIR / "train.csv.zip",
]
DATASET_DIR = BACKEND_DIR / "training_data" / "retail" / "dataset_products"
FULL_FRAME_BOX = "0.5 0.5 0.98 0.98"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
# train.csv headers vary between releases of this dataset, so the columns are matched by
# alias rather than by an exact name that a re-upload can quietly change.
IMAGE_ID_ALIASES = ("imgid", "image_id", "imageid", "image", "id", "filename")
CATEGORY_ALIASES = ("categories", "category", "label", "class", "class_name")


def _find_default_csv() -> Path:
    for candidate in DEFAULT_CSV_CANDIDATES:
        if candidate.exists():
            return candidate

    # Kaggle mounts a competition at /kaggle/input/<slug>, but the slug is not always the
    # competition's own name: this one arrives under /kaggle/input/competitions/. Search the
    # mount instead of guessing at its layout.
    if KAGGLE_INPUT_DIR.is_dir():
        for name in ("train.csv", "train.csv.zip"):
            found = sorted(KAGGLE_INPUT_DIR.rglob(name))
            if found:
                return found[0]

    searched = ", ".join(str(candidate) for candidate in DEFAULT_CSV_CANDIDATES)
    raise SystemExit(
        "Could not find train.csv. Pass --csv explicitly, or (on Kaggle) add the "
        "'retail-products-classification' competition as a data source.\n"
        f"Looked at {searched}, and searched everything under {KAGGLE_INPUT_DIR}."
    )


def _holds_images(directory: Path) -> bool:
    """True if the directory directly contains at least one image file."""
    try:
        return any(
            entry.suffix.lower() in IMAGE_EXTENSIONS and entry.is_file()
            for entry in directory.iterdir()
        )
    except OSError:
        return False


def _find_default_images_dir(csv_path: Path) -> Path:
    """Finds the product images belonging to train.csv, looking beside the csv itself.

    Resolved relative to the csv for the same reason _find_default_csv searches rather
    than hardcodes: the competition's mount point is not known ahead of time.
    """
    root = csv_path.parent
    candidates = [root / "train", root / "images", root / "train_images", root]
    if root.is_dir():
        candidates += [entry for entry in sorted(root.iterdir()) if entry.is_dir()]

    for candidate in candidates:
        if candidate.is_dir() and _holds_images(candidate):
            return candidate

    raise SystemExit(
        f"Found {csv_path} but no directory of images beside it. "
        "Pass --images-dir to point at wherever the images actually live."
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


def _resolve_columns(fieldnames: list[str] | None) -> tuple[str, str]:
    """Finds the image-id and category columns, tolerating case and spacing."""
    lookup = {(name or "").strip().lower().replace(" ", "_"): name for name in fieldnames or []}
    image_column = next((lookup[alias] for alias in IMAGE_ID_ALIASES if alias in lookup), None)
    category_column = next((lookup[alias] for alias in CATEGORY_ALIASES if alias in lookup), None)
    if image_column is None or category_column is None:
        raise SystemExit(
            "train.csv needs an image-id column and a category column. "
            f"Found columns: {list(lookup.values())}"
        )
    return image_column, category_column


def _normalize_category(raw: str) -> str:
    """Canonical class name for a raw CSV category value.

    Only case and spacing are normalized. Category strings that look hierarchical
    ("Beauty > Hair Care") are left whole deliberately: splitting them would be a guess
    about this dataset's taxonomy, and _clean_labels reports every value that merged so
    a real hierarchy shows up in the output instead of being silently flattened.
    """
    return _slugify(re.sub(r"\s+", " ", (raw or "").strip()))


def _clean_labels(rows: list[dict[str, str]]) -> tuple[list[tuple[str, str]], dict]:
    """Turns raw csv rows into (image_id, class_name) pairs, dropping what can't be trained on.

    Removes blank ids and labels, collapses duplicate rows, and drops any image whose rows
    disagree about its category, since an ambiguous label is worse than a missing one.
    """
    image_column, category_column = _resolve_columns(list(rows[0]) if rows else None)

    blank_id = blank_category = 0
    labels: dict[str, set[str]] = collections.defaultdict(set)
    raw_by_canonical: dict[str, set[str]] = collections.defaultdict(set)
    duplicate_rows = 0
    seen: set[tuple[str, str]] = set()

    for row in rows:
        image_id = (row.get(image_column) or "").strip()
        raw_category = (row.get(category_column) or "").strip()
        if not image_id:
            blank_id += 1
            continue
        category = _normalize_category(raw_category)
        if not category:
            blank_category += 1
            continue
        if (image_id, category) in seen:
            duplicate_rows += 1
            continue
        seen.add((image_id, category))
        labels[image_id].add(category)
        raw_by_canonical[category].add(raw_category)

    conflicting = {image_id for image_id, names in labels.items() if len(names) > 1}
    pairs = [(image_id, next(iter(names))) for image_id, names in labels.items() if len(names) == 1]
    merged = {name: sorted(raws) for name, raws in raw_by_canonical.items() if len(raws) > 1}

    if blank_id:
        print(f"  dropped {blank_id:,} rows with a blank image id")
    if blank_category:
        print(f"  dropped {blank_category:,} rows with a blank category")
    if duplicate_rows:
        print(f"  collapsed {duplicate_rows:,} duplicate rows")
    if conflicting:
        print(f"  dropped {len(conflicting):,} images labelled with more than one category")
    if merged:
        print(f"  merged {len(merged):,} categories that differed only by case or spacing")
        for name, raws in sorted(merged.items())[:5]:
            print(f"    {name} <- {raws}")

    return pairs, {
        "image_column": image_column,
        "category_column": category_column,
        "blank_image_id": blank_id,
        "blank_category": blank_category,
        "duplicate_rows": duplicate_rows,
        "conflicting_labels": len(conflicting),
        "merged_categories": merged,
    }


def _content_hash(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_image(path: Path, min_size: int, verify: bool, want_hash: bool) -> tuple[str, str | None]:
    """Returns (status, content hash). status is 'ok' or the reason to drop this image."""
    try:
        if verify:
            with Image.open(path) as image:
                image.verify()  # cheap structural check; invalidates the handle, so reopen below
            with Image.open(path) as image:
                width, height = image.size
                if min(width, height) < min_size:
                    return f"smaller than {min_size}px", None
                image.load()  # full decode, catching truncation that verify() lets through
        return "ok", _content_hash(path) if want_hash else None
    except Exception as exc:
        return f"unreadable ({type(exc).__name__})", None


def _inspect_images(
    pairs: list[tuple[Path, str]], min_size: int, workers: int, verify: bool, dedupe: bool
) -> tuple[list[tuple[Path, str]], dict]:
    """Drops unreadable, undersized and byte-identical images, in parallel.

    Corrupt files matter because Ultralytics hits them mid-epoch, hours in. Duplicates
    matter because the train/val split is random: the same photo landing on both sides
    inflates val mAP for free. Matching is on exact bytes, so a re-encode of the same
    photo is not caught - that needs perceptual hashing and a threshold to tune.
    """
    if not verify and not dedupe:
        return pairs, {"verified": False, "deduped": False}

    kept: list[tuple[Path, str]] = []
    dropped: collections.Counter = collections.Counter()
    seen_hashes: dict[str, Path] = {}
    duplicates = 0

    print(f"\nInspecting {len(pairs):,} images ({workers} workers)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        statuses = pool.map(
            lambda pair: _inspect_image(pair[0], min_size, verify, dedupe), pairs, chunksize=64
        )
        for (path, class_name), (status, content_hash) in zip(pairs, statuses):
            if status != "ok":
                dropped[status] += 1
                continue
            if dedupe and content_hash is not None:
                if content_hash in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes[content_hash] = path
            kept.append((path, class_name))

    for reason, count in dropped.most_common():
        print(f"  dropped {count:,} images: {reason}")
    if duplicates:
        print(f"  dropped {duplicates:,} byte-identical duplicate images")
    print(f"  {len(kept):,} images usable")

    return kept, {
        "verified": verify,
        "deduped": dedupe,
        "dropped_by_reason": dict(dropped),
        "duplicate_images": duplicates,
        "usable": len(kept),
    }


def _write_image(src: Path, dest_dir: Path, stem: str) -> None:
    """Copies an image into the dataset, normalizing anything that isn't already RGB.

    Grayscale, palette, CMYK and RGBA sources otherwise reach the trainer with a channel
    count it has to reconcile per batch; re-encoding only those keeps the common case a
    plain byte copy.
    """
    try:
        with Image.open(src) as image:
            if image.mode == "RGB":
                shutil.copy(src, dest_dir / f"{stem}{src.suffix}")
            else:
                image.convert("RGB").save(dest_dir / f"{stem}.jpg", format="JPEG", quality=95)
    except Exception:
        shutil.copy(src, dest_dir / f"{stem}{src.suffix}")


def _cap_per_class(by_class: dict[str, list[Path]], cap: int, seed: int) -> int:
    """Trims over-represented classes so one huge category can't dominate training."""
    rng = random.Random(seed)
    removed = 0
    for class_name, paths in by_class.items():
        if len(paths) > cap:
            rng.shuffle(paths)
            removed += len(paths) - cap
            by_class[class_name] = paths[:cap]
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=None, help="Path to train.csv (or train.csv.zip); auto-detected if omitted")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing the product images, named <ImgId>.<ext>; "
        "auto-detected beside train.csv if omitted",
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
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap images kept per category, so one over-represented class can't dominate",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=32,
        help="Drop images whose shorter side is below this many pixels",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip decoding every image to check it isn't corrupt (faster, riskier)",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep byte-identical duplicate images instead of dropping them. They leak "
        "across the train/val split and inflate val mAP, so this is rarely what you want",
    )
    parser.add_argument("--workers", type=int, default=8, help="Threads used to inspect images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = args.csv or _find_default_csv()
    print(f"Reading labels from {csv_path}")
    rows = _read_rows(csv_path)
    print(f"  {len(rows):,} rows")

    images_dir = args.images_dir or _find_default_images_dir(csv_path)
    print(f"Reading images from {images_dir}")

    if not images_dir.exists():
        raise SystemExit(
            f"Missing images directory {images_dir}. On Kaggle, add the "
            "'retail-products-classification' competition as a data source to this notebook, "
            "or pass --images-dir to point at wherever the images actually live."
        )

    print("\nCleaning labels...")
    labelled, label_report = _clean_labels(rows)
    print(f"  {len(labelled):,} images with a usable label")

    resolved: list[tuple[Path, str]] = []
    missing_files = 0
    for image_id, class_name in labelled:
        image_path = _resolve_image(images_dir, image_id)
        if image_path is None:
            missing_files += 1
            continue
        resolved.append((image_path, class_name))

    resolved, image_report = _inspect_images(
        resolved,
        min_size=args.min_image_size,
        workers=args.workers,
        verify=not args.skip_verify,
        dedupe=not args.keep_duplicates,
    )

    by_class: dict[str, list[Path]] = {}
    for image_path, class_name in resolved:
        by_class.setdefault(class_name, []).append(image_path)

    capped_per_class = 0
    if args.max_per_class is not None:
        capped_per_class = _cap_per_class(by_class, args.max_per_class, args.seed)
        if capped_per_class:
            print(f"\nTrimmed {capped_per_class:,} images from classes over --max-per-class {args.max_per_class}")

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
        print(f"{missing_files:,} rows skipped: image file not found in {images_dir}")

    _reset_dataset_dir()
    class_ids = {name: i for i, name in enumerate(class_names)}
    rng = random.Random(args.seed)

    total_train = total_val = 0
    for class_name, paths in by_class.items():
        class_id = class_ids[class_name]
        rng.shuffle(paths)
        split_at = max(1, int(len(paths) * (1 - args.val_fraction)))

        for index, src in enumerate(paths):
            split = "train" if index < split_at else "val"
            dest_name = f"{_slugify(class_name)}_{src.stem}"
            _write_image(src, DATASET_DIR / "images" / split, dest_name)
            (DATASET_DIR / "labels" / split / f"{dest_name}.txt").write_text(
                f"{class_id} {FULL_FRAME_BOX}\n", encoding="utf-8"
            )
            if split == "train":
                total_train += 1
            else:
                total_val += 1

    data_yaml_path = _write_data_yaml(class_names)
    (DATASET_DIR / "classes.json").write_text(json.dumps(class_ids, indent=2), encoding="utf-8")

    counts = sorted((len(paths) for paths in by_class.values()), reverse=True)
    report = {
        "csv": str(csv_path),
        "images_dir": str(images_dir),
        "rows_read": len(rows),
        "labels": label_report,
        "images": image_report,
        "missing_files": missing_files,
        "capped_per_class": capped_per_class,
        "classes_dropped_too_few_images": len(dropped),
        "classes": len(class_names),
        "train_images": total_train,
        "val_images": total_val,
        "largest_class": counts[0] if counts else 0,
        "smallest_class": counts[-1] if counts else 0,
    }
    (DATASET_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nDataset ready: {total_train:,} train / {total_val:,} val images across {len(class_names)} classes")
    if counts:
        print(f"Class sizes: largest {counts[0]:,}, median {counts[len(counts) // 2]:,}, smallest {counts[-1]:,}")
        if counts[0] > 20 * counts[-1]:
            print("  Class imbalance is steep - consider --max-per-class to trim the biggest categories.")
    print(f"data.yaml written to {data_yaml_path}")
    print(f"cleaning report written to {DATASET_DIR / 'cleaning_report.json'}")


if __name__ == "__main__":
    main()
