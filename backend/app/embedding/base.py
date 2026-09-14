from abc import ABC, abstractmethod


class ImageEmbedder(ABC):
    """Port for turning an image into a fixed-length vector for similarity search."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, image_bytes: bytes) -> list[float]:
        raise NotImplementedError
