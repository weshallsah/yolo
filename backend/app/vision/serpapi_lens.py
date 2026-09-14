import io

import requests
from PIL import Image

from app.vision.base import LensMatch, ReverseImageSearcher, ReverseImageSearchError

_MAX_UPLOAD_BYTES = 500_000


def _shrink_to_upload_limit(image_bytes: bytes) -> bytes:
    """Re-encodes the image as JPEG under SerpApi's 500KB upload limit."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_dimension = 1280

    while True:
        working = image.copy()
        working.thumbnail((max_dimension, max_dimension))

        for quality in (85, 70, 55, 40):
            buffer = io.BytesIO()
            working.save(buffer, format="JPEG", quality=quality)
            encoded = buffer.getvalue()
            if len(encoded) <= _MAX_UPLOAD_BYTES:
                return encoded

        if max_dimension <= 320:
            return encoded
        max_dimension = int(max_dimension * 0.75)


class SerpApiLensSearcher(ReverseImageSearcher):
    """Identifies an unknown object via SerpApi's Google Lens reverse-image search."""

    def __init__(self, api_key: str, base_url: str, timeout: float = 60.0) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

    def search(self, image_bytes: bytes) -> LensMatch | None:
        if not self._api_key:
            raise ReverseImageSearchError(
                "APP_SERPAPI_KEY is not configured; reverse image search is unavailable."
            )

        image_id = self._upload_image(image_bytes)
        return self._lookup_visual_match(image_id)

    def _upload_image(self, image_bytes: bytes) -> str:
        payload = _shrink_to_upload_limit(image_bytes)

        try:
            response = requests.post(
                f"{self._base_url}/image",
                files={"image": ("query.jpg", payload, "image/jpeg")},
                data={"api_key": self._api_key},
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as error:
            raise ReverseImageSearchError(f"Image upload to Google Lens failed: {error}") from error

        image_id = body.get("image_id")
        if not image_id:
            raise ReverseImageSearchError("Google Lens image upload did not return an image_id.")
        return image_id

    def _lookup_visual_match(self, image_id: str) -> LensMatch | None:
        try:
            response = requests.get(
                f"{self._base_url}/search",
                params={
                    "engine": "google_lens",
                    "image_id": image_id,
                    "type": "visual_matches",
                    "api_key": self._api_key,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            raise ReverseImageSearchError(f"Google Lens search failed: {error}") from error

        matches = payload.get("visual_matches", [])
        if not matches:
            return None

        best = matches[0]
        title = best.get("title")
        if not title:
            return None

        return LensMatch(
            title=title,
            source=best.get("source", "Google Lens"),
            url=best.get("link", ""),
            thumbnail=(best.get("thumbnail") or {}).get("url")
            if isinstance(best.get("thumbnail"), dict)
            else best.get("thumbnail"),
        )
