import io
import logging
import re
import uuid
from pathlib import Path

from PIL import Image

from app.training.base import TrainingDataRecorder

logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "object"


class YoloDatasetRecorder(TrainingDataRecorder):
    """Accumulates a pool of new-class images for later YOLO fine-tuning.

    Every object this app can't detect with YOLO but can name via Google Lens gets its
    source image saved here, one folder per generic class name. This is a raw image
    pool, not a ready-to-train dataset: `scripts/train_yolo.py` assigns real class ids,
    generates YOLO-format labels, and builds the actual train/val split from it.

    Recording failures are logged and swallowed rather than raised, since this is a
    best-effort side channel and must never break the user-facing identify request.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def record(self, image_bytes: bytes, class_name: str) -> None:
        try:
            class_dir = self._root / "pool" / _slugify(class_name)
            class_dir.mkdir(parents=True, exist_ok=True)

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.save(class_dir / f"{uuid.uuid4().hex}.jpg", format="JPEG", quality=90)
        except Exception:
            logger.warning("Failed to record training sample for class %r", class_name, exc_info=True)
