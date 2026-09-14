from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.db.base import get_session_factory
from app.detection.base import ObjectDetector
from app.detection.yolo_detector import YoloObjectDetector
from app.embedding.base import ImageEmbedder
from app.embedding.clip_embedder import ClipImageEmbedder
from app.repositories.base import ProductCatalogRepository
from app.repositories.pgvector_repository import PgVectorProductRepository
from app.search.base import ProductSearcher
from app.search.serpapi_shopping import SerpApiShoppingSearcher
from app.services.identify_service import IdentifyService
from app.training.base import TrainingDataRecorder
from app.training.dataset_recorder import YoloDatasetRecorder
from app.vision.base import ReverseImageSearcher
from app.vision.serpapi_lens import SerpApiLensSearcher


@lru_cache
def get_detector() -> ObjectDetector:
    settings = get_settings()
    return YoloObjectDetector(
        weights_path=settings.yolo_weights_path,
        confidence_threshold=settings.detection_confidence_threshold,
    )


@lru_cache
def get_embedder() -> ImageEmbedder:
    return ClipImageEmbedder()


@lru_cache
def get_catalog_repository() -> ProductCatalogRepository:
    settings = get_settings()
    return PgVectorProductRepository(
        session_factory=get_session_factory(),
        max_distance=settings.visual_match_max_distance,
    )


@lru_cache
def get_reverse_image_searcher() -> ReverseImageSearcher:
    settings = get_settings()
    return SerpApiLensSearcher(api_key=settings.serpapi_key, base_url=settings.serpapi_base_url)


@lru_cache
def get_product_searcher() -> ProductSearcher:
    settings = get_settings()
    return SerpApiShoppingSearcher(
        api_key=settings.serpapi_key,
        base_url=settings.serpapi_base_url,
        country=settings.shopping_country,
        language=settings.shopping_language,
        currency=settings.shopping_currency,
    )


@lru_cache
def get_training_recorder() -> TrainingDataRecorder:
    settings = get_settings()
    return YoloDatasetRecorder(root=Path(settings.training_data_dir))


def get_identify_service() -> IdentifyService:
    settings = get_settings()
    return IdentifyService(
        embedder=get_embedder(),
        catalog=get_catalog_repository(),
        detector=get_detector(),
        reverse_search=get_reverse_image_searcher(),
        product_search=get_product_searcher(),
        training_recorder=get_training_recorder(),
        yolo_trust_confidence=settings.yolo_trust_confidence,
    )
