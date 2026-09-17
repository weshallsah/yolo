import argparse
from pathlib import Path

from ultralytics import YOLO

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_YAML_PATH = BACKEND_DIR / "training_data" / "retail" / "dataset_products" / "data.yaml"
MODELS_DIR = BACKEND_DIR / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yolov8n.pt", help="Base checkpoint to start training from")
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
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="DataLoader worker processes. Use 0 on environments with restricted shared memory, "
        "where workers>0 can deadlock.",
    )
    parser.add_argument("--patience", type=int, default=10, help="Stop early once val mAP plateaus for this many epochs")
    parser.add_argument(
        "--cache",
        choices=["ram", "disk", "none"],
        default="ram",
        help="Image caching strategy: ram (fastest, falls back to none if there isn't enough RAM), "
        "disk (caches resized images as .npy files, needs less RAM but more disk space), "
        "or none (reads from disk every batch)",
    )
    args = parser.parse_args()

    if not DATA_YAML_PATH.exists():
        raise SystemExit(f"Missing {DATA_YAML_PATH}. Run scripts/prepare_retail_dataset.py first.")

    cache = {"ram": True, "disk": "disk", "none": False}[args.cache]
    batch = int(args.batch) if args.batch >= 1 else args.batch

    model = YOLO(args.weights)
    model.train(
        data=str(DATA_YAML_PATH),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        cache=cache,
        project=str(MODELS_DIR),
        name="retail_yolo_products",
        exist_ok=True,
    )

    print(
        "\nDone. Check the validation metrics above, then point APP_YOLO_WEIGHTS_PATH at "
        f"{MODELS_DIR / 'retail_yolo_products' / 'weights' / 'best.pt'} to start serving it."
    )


if __name__ == "__main__":
    main()
