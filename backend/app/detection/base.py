from abc import ABC, abstractmethod

from app.schemas import Detection


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, image_bytes: bytes) -> list[Detection]:
        raise NotImplementedError
