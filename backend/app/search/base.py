from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ShoppingListing:
    """A single real, live marketplace listing for a product."""

    title: str
    source: str
    price: float
    currency: str
    url: str
    thumbnail: str | None
    condition: str | None


class ProductSearcher(ABC):
    """Port for finding real, live marketplace listings for a product query."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[ShoppingListing]:
        raise NotImplementedError


class ProductSearchError(Exception):
    """Raised when the live product search backend can't be reached or errors out."""
