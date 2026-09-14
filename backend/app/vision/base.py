from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LensMatch:
    """The best real visual match for a query image, from a reverse-image search."""

    title: str
    source: str
    url: str
    thumbnail: str | None


class ReverseImageSearcher(ABC):
    """Port for identifying an object by reverse image search when detection fails."""

    @abstractmethod
    def search(self, image_bytes: bytes) -> LensMatch | None:
        raise NotImplementedError


class ReverseImageSearchError(Exception):
    """Raised when the reverse image search backend can't be reached or errors out."""
