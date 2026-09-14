from abc import ABC, abstractmethod


class TrainingDataRecorder(ABC):
    """Port for recording a new-class training sample (image + generic object label)."""

    @abstractmethod
    def record(self, image_bytes: bytes, class_name: str) -> None:
        raise NotImplementedError
