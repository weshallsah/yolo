import uuid

from app.detection.base import ObjectDetector
from app.embedding.base import ImageEmbedder
from app.repositories.base import ProductCatalogRepository
from app.schemas import IdentifyResult, PriceListing, Trust
from app.search.base import ProductSearcher, ProductSearchError, ShoppingListing
from app.search.pricing import average_price, best_value
from app.search.query_broadening import broadened_queries
from app.services.exceptions import IdentificationError
from app.vision.base import ReverseImageSearcher, ReverseImageSearchError


def _trust_from_confidence(confidence: float, source: str) -> Trust:
    """Derives a trust assessment from the real confidence/similarity score of the match."""
    if confidence >= 0.85:
        level = "verified"
        note = "high-confidence"
    elif confidence >= 0.5:
        level = "caution"
        note = "moderate-confidence"
    else:
        level = "unverified"
        note = "low-confidence"

    return Trust(
        level=level,
        reason=f"{note.capitalize()} {source} match ({confidence:.0%}); "
        "exact brand/model isn't independently confirmed.",
    )


def _to_price_listing(listing: ShoppingListing, is_best: bool) -> PriceListing:
    return PriceListing(
        source=listing.source,
        price=listing.price,
        currency=listing.currency,
        condition=listing.condition or "Unknown",
        url=listing.url,
        thumbnail=listing.thumbnail,
        is_best_value=is_best,
    )


class IdentifyService:
    """Orchestrates detection and live enrichment to answer 'what is this, and is it worth it?'.

    Flow: check the pgvector catalog for a visual match first; if nothing is close
    enough, try YOLO object detection, but only trust it when confident (YOLO is a
    closed-set detector limited to ~80 COCO classes, so a low-confidence guess is
    usually a wrong forced fit, not a real match); anything YOLO can't confidently
    name falls back to a real reverse-image search (Google Lens via SerpApi), which
    can recognize open-vocabulary/specific products YOLO was never trained on. Every
    YOLO- or Lens-derived identification is written back into the vector catalog so
    the same object is matched directly next time, without needing detection or Lens
    again.
    """

    def __init__(
        self,
        embedder: ImageEmbedder,
        catalog: ProductCatalogRepository,
        detector: ObjectDetector,
        reverse_search: ReverseImageSearcher,
        product_search: ProductSearcher,
        yolo_trust_confidence: float = 0.6,
    ) -> None:
        self._embedder = embedder
        self._catalog = catalog
        self._detector = detector
        self._reverse_search = reverse_search
        self._product_search = product_search
        self._yolo_trust_confidence = yolo_trust_confidence

    def identify(self, image_bytes: bytes) -> IdentifyResult:
        embedding = self._embedder.embed(image_bytes)

        match = self._catalog.find_closest(embedding)
        if match is not None:
            return self._build_result(
                title=match.title,
                category=match.category,
                description=match.description,
                confidence=match.similarity,
                source="catalog",
            )

        detections = self._detector.detect(image_bytes)
        if detections and detections[0].confidence >= self._yolo_trust_confidence:
            label = detections[0].label
            title = label.title()
            category = label.title()
            description = f"Identified as '{label}' by object detection."
            confidence = detections[0].confidence
            source = "YOLO detection"
            self._catalog.add(
                embedding=embedding,
                title=title,
                category=category,
                description=description,
                trust=_trust_from_confidence(confidence, source),
            )
            return self._build_result(
                title=title,
                category=category,
                description=description,
                confidence=confidence,
                source=source,
            )

        try:
            lens_match = self._reverse_search.search(image_bytes)
        except ReverseImageSearchError as error:
            raise IdentificationError(str(error)) from error

        if lens_match is None:
            raise IdentificationError("Could not identify a product in this image.")

        title = lens_match.title
        category = "Identified via Google Lens"
        description = f"Matched via Google Lens reverse image search (source: {lens_match.source})."
        confidence = 0.6
        source = "Google Lens"
        self._catalog.add(
            embedding=embedding,
            title=title,
            category=category,
            description=description,
            trust=_trust_from_confidence(confidence, source),
        )
        return self._build_result(
            title=title,
            category=category,
            description=description,
            confidence=confidence,
            source=source,
        )

    def _search_with_fallback(self, title: str) -> tuple[list[ShoppingListing], str]:
        """Searches by the exact identified title, then by progressively more generic
        category terms, so a too-specific identification (e.g. an exact color/size
        variant) still surfaces live listings for the general object.
        """
        queries = [title, *broadened_queries(title)]
        seen: set[str] = set()
        last_error: ProductSearchError | None = None

        for query in queries:
            if query.lower() in seen:
                continue
            seen.add(query.lower())

            try:
                listings = self._product_search.search(query)
            except ProductSearchError as error:
                last_error = error
                continue

            if listings:
                return listings, query

        if last_error is not None:
            raise IdentificationError(str(last_error))
        raise IdentificationError(
            f"Identified '{title}' but found no live pricing data for it right now."
        )

    def _build_result(
        self,
        *,
        title: str,
        category: str,
        description: str,
        confidence: float,
        source: str,
    ) -> IdentifyResult:
        listings, matched_query = self._search_with_fallback(title)

        avg_price = average_price(listings)
        cheapest = best_value(listings)
        price_listings = [
            _to_price_listing(listing, is_best=listing is cheapest) for listing in listings
        ]
        trust = _trust_from_confidence(confidence, source)

        recommendation = (
            f"This looks like a {title.lower()}. The best live price found is from "
            f"{cheapest.source} at ${cheapest.price:.2f}, compared to a live market average of "
            f"${avg_price:.2f}. Trust assessment: {trust.reason}"
        )
        if matched_query.lower() != title.lower():
            recommendation = (
                f"No live listings matched the exact identified product, so these are listings "
                f"for the general category '{matched_query}' instead. " + recommendation
            )

        return IdentifyResult(
            id=str(uuid.uuid4()),
            title=title,
            category=category,
            description=description,
            confidence=confidence,
            trust=trust,
            price_listings=price_listings,
            average_price=avg_price,
            recommendation=recommendation,
        )
