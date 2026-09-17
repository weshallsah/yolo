import argparse
from pathlib import Path

from ultralytics import YOLO

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_YAML_PATH = BACKEND_DIR / "training_data" / "abo" / "dataset_electronics" / "data.yaml"
MODELS_DIR = BACKEND_DIR / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yolov8n.pt", help="Base checkpoint to start training from")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. 0 for first GPU, cpu for CPU (default: auto)")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="DataLoader worker processes. Use 0 on environments with restricted shared memory (e.g. Kaggle), "
        "where workers>0 can deadlock.",
    )
    parser.add_argument("--patience", type=int, default=10, help="Stop early once val mAP plateaus for this many epochs")
    args = parser.parse_args()

    if not DATA_YAML_PATH.exists():
        raise SystemExit(f"Missing {DATA_YAML_PATH}. Run scripts/prepare_abo_dataset.py first.")

    model = YOLO(args.weights)
    model.train(
        data=str(DATA_YAML_PATH),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        cache=True,
        project=str(MODELS_DIR),
        name="abo_yolo_electronics",
        exist_ok=True,
    )

    print(
        "\nDone. Check the validation metrics above, then point APP_YOLO_WEIGHTS_PATH at "
        f"{MODELS_DIR / 'abo_yolo_electronics' / 'weights' / 'best.pt'} to start serving it."
    )


if __name__ == "__main__":
    main()
