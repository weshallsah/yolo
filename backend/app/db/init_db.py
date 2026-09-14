from sqlalchemy import text

from app.db.base import Base, get_engine


def init_db() -> None:
    """Ensures the pgvector extension and schema exist. Safe to call on every startup."""
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
