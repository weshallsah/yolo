from functools import lru_cache

from app.config import get_settings
from app.detection.base import ObjectDetector
from app.detection.yolo_detector import YoloObjectDetector


@lru_cache
def get_detector() -> ObjectDetector:
    settings = get_settings()
    return YoloObjectDetector(
        weights_path=settings.yolo_weights_path,
        confidence_threshold=settings.detection_confidence_threshold,
    )
