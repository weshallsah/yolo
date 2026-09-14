from abc import ABC, abstractmethod

from app.schemas import Detection


class ObjectDetector(ABC):
    """Port for anything that can locate and label objects in an image."""

    @abstractmethod
    def detect(self, image_bytes: bytes) -> list[Detection]:
        """Return detections found in the image, ordered by confidence (highest first)."""
        raise NotImplementedError
