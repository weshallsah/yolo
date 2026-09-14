from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas import Trust


@dataclass(frozen=True)
class CatalogMatch:
    """A catalog product found to be visually similar to a query image."""

    title: str
    category: str
    description: str
    trust: Trust
    similarity: float


class ProductCatalogRepository(ABC):
    """Port for finding and learning products in the visual similarity catalog."""

    @abstractmethod
    def find_closest(self, embedding: list[float]) -> CatalogMatch | None:
        raise NotImplementedError

    @abstractmethod
    def add(
        self,
        embedding: list[float],
        title: str,
        category: str,
        description: str,
        trust: Trust,
    ) -> None:
        """Stores a newly identified product so it can be matched directly next time."""
        raise NotImplementedError
