from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Product
from app.repositories.base import CatalogMatch, ProductCatalogRepository
from app.schemas import Trust


class PgVectorProductRepository(ProductCatalogRepository):
    """Finds and learns catalog products via pgvector cosine distance."""

    def __init__(self, session_factory: Callable[[], Session], max_distance: float) -> None:
        self._session_factory = session_factory
        self._max_distance = max_distance

    def find_closest(self, embedding: list[float]) -> CatalogMatch | None:
        distance = Product.embedding.cosine_distance(embedding)

        with self._session_factory() as session:
            row = session.execute(
                select(Product, distance.label("distance")).order_by(distance).limit(1)
            ).first()

        if row is None:
            return None

        product, distance_value = row
        if distance_value > self._max_distance:
            return None

        return CatalogMatch(
            title=product.title,
            category=product.category,
            description=product.description,
            trust=Trust(level=product.trust_level, reason=product.trust_reason),
            similarity=1 - distance_value,
        )

    def add(
        self,
        embedding: list[float],
        title: str,
        category: str,
        description: str,
        trust: Trust,
    ) -> None:
        with self._session_factory() as session:
            session.add(
                Product(
                    title=title,
                    category=category,
                    description=description,
                    trust_level=trust.level,
                    trust_reason=trust.reason,
                    embedding=embedding,
                )
            )
            session.commit()
