"""Clears every learned product from the pgvector catalog, keeping the schema intact.

Run with: python -m scripts.clear_catalog
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.base import Base, get_engine  # noqa: E402
from app.db.init_db import init_db  # noqa: E402
from app.db.models import Product  # noqa: E402


def clear() -> None:
    """Drops and recreates the products table, so schema changes are picked up too."""
    init_db()
    engine = get_engine()
    Product.__table__.drop(engine, checkfirst=True)
    Base.metadata.create_all(engine)
    print("Catalog cleared: products table dropped and recreated empty.")
    engine.dispose()


if __name__ == "__main__":
    clear()
