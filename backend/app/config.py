import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, overridable via environment variables."""

    app_name: str = "Product Identifier API"
    host: str = "0.0.0.0"
    port: int = 9000
    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    yolo_weights_path: str = "yolo11n.pt"
    detection_confidence_threshold: float = 0.2
    yolo_trust_confidence: float = 0.6
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/productdb"
    visual_match_max_distance: float = 0.22
    serpapi_key: str = ""
    serpapi_base_url: str = "https://serpapi.com"
    shopping_country: str = "us"
    shopping_language: str = "en"
    shopping_currency: str = "USD"
    training_data_dir: str = "training_data"


@lru_cache
def get_settings() -> Settings:
    port = int(os.environ.get("APP_PORT", 9000))
    weights_path = os.environ.get("APP_YOLO_WEIGHTS_PATH", "yolo11n.pt")
    threshold = float(os.environ.get("APP_DETECTION_CONFIDENCE_THRESHOLD", 0.2))
    yolo_trust_confidence = float(os.environ.get("APP_YOLO_TRUST_CONFIDENCE", 0.6))
    extra_origin = os.environ.get("APP_CORS_ORIGIN")
    database_url = os.environ.get(
        "APP_DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/productdb"
    )
    visual_match_max_distance = float(os.environ.get("APP_VISUAL_MATCH_MAX_DISTANCE", 0.22))
    serpapi_key = os.environ.get("APP_SERPAPI_KEY", "")
    serpapi_base_url = os.environ.get("APP_SERPAPI_BASE_URL", "https://serpapi.com")
    shopping_country = os.environ.get("APP_SHOPPING_COUNTRY", "us")
    shopping_language = os.environ.get("APP_SHOPPING_LANGUAGE", "en")
    shopping_currency = os.environ.get("APP_SHOPPING_CURRENCY", "USD")
    training_data_dir = os.environ.get("APP_TRAINING_DATA_DIR", "training_data")

    cors_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    if extra_origin:
        cors_origins.append(extra_origin)

    return Settings(
        port=port,
        yolo_weights_path=weights_path,
        detection_confidence_threshold=threshold,
        yolo_trust_confidence=yolo_trust_confidence,
        cors_origins=cors_origins,
        database_url=database_url,
        visual_match_max_distance=visual_match_max_distance,
        serpapi_key=serpapi_key,
        serpapi_base_url=serpapi_base_url,
        shopping_country=shopping_country,
        shopping_language=shopping_language,
        shopping_currency=shopping_currency,
        training_data_dir=training_data_dir,
    )
