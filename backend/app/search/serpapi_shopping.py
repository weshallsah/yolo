import requests

from app.search.base import ProductSearcher, ProductSearchError, ShoppingListing


class SerpApiShoppingSearcher(ProductSearcher):
    """Finds real, live marketplace listings via SerpApi's Google Shopping engine."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        country: str,
        language: str,
        currency: str,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._country = country
        self._language = language
        self._currency = currency
        self._timeout = timeout

    def search(self, query: str, limit: int = 10) -> list[ShoppingListing]:
        if not self._api_key:
            raise ProductSearchError(
                "APP_SERPAPI_KEY is not configured; live pricing lookups are unavailable."
            )

        try:
            response = requests.get(
                f"{self._base_url}/search",
                params={
                    "engine": "google_shopping",
                    "q": query,
                    "gl": self._country,
                    "hl": self._language,
                    "api_key": self._api_key,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            raise ProductSearchError(f"Live product search failed: {error}") from error

        results = payload.get("shopping_results", [])
        listings: list[ShoppingListing] = []

        for result in results[:limit]:
            price = result.get("extracted_price")
            title = result.get("title")
            source = result.get("source")
            if price is None or not title or not source:
                continue

            listings.append(
                ShoppingListing(
                    title=title,
                    source=source,
                    price=float(price),
                    currency=self._currency,
                    url=result.get("product_link") or result.get("link") or "",
                    thumbnail=result.get("thumbnail"),
                    condition=result.get("condition"),
                )
            )

        return listings
