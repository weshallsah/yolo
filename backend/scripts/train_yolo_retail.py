import argparse
from pathlib import Path

from ultralytics import YOLO

from prepare_retail_dataset import DATASET_DIRS

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR / "models"
DEFAULT_WEIGHTS = {"classify": "yolov8n-cls.pt", "detect": "yolov8n.pt"}
RUN_NAMES = {"classify": "retail_yolo_products_cls", "detect": "retail_yolo_products"}
# Classifiers train on square crops and gain little from detection's 640px, so each task
# gets the resolution its head was designed around unless --imgsz overrides it.
DEFAULT_IMGSZ = {"classify": 224, "detect": 640}


def _resolve_data(task: str) -> str:
    """What Ultralytics trains on: a data.yaml for detect, the dataset root for classify."""
    dataset_dir = DATASET_DIRS[task]
    if task == "detect":
        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.exists():
            raise SystemExit(
                f"Missing {data_yaml}. Run "
                "scripts/prepare_retail_dataset.py --layout detect first."
            )
        return str(data_yaml)

    if not (dataset_dir / "train").is_dir():
        raise SystemExit(
            f"Missing {dataset_dir / 'train'}. Run "
            "scripts/prepare_retail_dataset.py --layout classify first."
        )
    return str(dataset_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task",
        choices=sorted(DATASET_DIRS),
        default="classify",
        help="classify: train yolov8n-cls on this dataset's image-level labels. "
        "detect: train a detector on the full-frame-box shim",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Base checkpoint to start training from (default: yolov8n-cls.pt for classify, yolov8n.pt for detect)",
    )
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
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="DataLoader worker processes. Use 0 on environments with restricted shared memory, "
        "where workers>0 can deadlock.",
    )
    parser.add_argument("--patience", type=int, default=10, help="Stop early once the val metric plateaus for this many epochs")
    parser.add_argument(
        "--cache",
        choices=["ram", "disk", "none"],
        default="ram",
        help="Image caching strategy: ram (fastest, falls back to none if there isn't enough RAM), "
        "disk (caches resized images as .npy files, needs less RAM but more disk space), "
        "or none (reads from disk every batch)",
    )
    args = parser.parse_args()

    data = _resolve_data(args.task)
    weights = args.weights or DEFAULT_WEIGHTS[args.task]
    imgsz = args.imgsz if args.imgsz is not None else DEFAULT_IMGSZ[args.task]
    cache = {"ram": True, "disk": "disk", "none": False}[args.cache]
    batch = int(args.batch) if args.batch >= 1 else args.batch

    model = YOLO(weights)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        cache=cache,
        project=str(MODELS_DIR),
        name=RUN_NAMES[args.task],
        exist_ok=True,
    )

    best = MODELS_DIR / RUN_NAMES[args.task] / "weights" / "best.pt"
    metric = "top-1/top-5 accuracy" if args.task == "classify" else "mAP"
    print(f"\nDone. Check the validation {metric} above; best weights are at {best}.")
    if args.task == "classify":
        print(
            "This is a classifier, so the detection API in app/detection/yolo_detector.py "
            "cannot serve it. Point APP_YOLO_WEIGHTS_PATH at it and set APP_MODEL_TASK=classify."
        )
    else:
        print("Point APP_YOLO_WEIGHTS_PATH at it to start serving it.")


if __name__ == "__main__":
    main()
