"""Trains a YOLO model on the ABO product-catalog dataset built by prepare_abo_dataset.py.

Every class here is a product_type from ABO (e.g. "SHOES", "CHAIR") labeled with a
full-frame box, since ABO has no real bounding-box annotations (see
prepare_abo_dataset.py for why). This trains a standalone detector over ABO's classes
rather than fine-tuning the app's existing COCO-based checkpoint, since ABO's classes
mostly don't overlap with COCO's 80 and there's no need to preserve them here.

Usage:
    python scripts/train_yolo_abo.py [--weights yolo11n.pt] [--epochs 30] [--imgsz 640]

After training, point APP_YOLO_WEIGHTS_PATH at models/abo_yolo/weights/best.pt to serve it.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_YAML_PATH = BACKEND_DIR / "training_data" / "abo" / "dataset" / "data.yaml"
MODELS_DIR = BACKEND_DIR / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yolo11n.pt", help="Base checkpoint to start training from")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. 0 for first GPU, cpu for CPU (default: auto)")
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
        project=str(MODELS_DIR),
        name="abo_yolo",
        exist_ok=True,
    )

    print(
        "\nDone. Check the validation metrics above, then point APP_YOLO_WEIGHTS_PATH at "
        f"{MODELS_DIR / 'abo_yolo' / 'weights' / 'best.pt'} to start serving it."
    )


if __name__ == "__main__":
    main()
