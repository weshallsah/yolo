import io

import open_clip
import torch
from PIL import Image

from app.embedding.base import ImageEmbedder

_MODEL_NAME = "ViT-B-32-quickgelu"
_PRETRAINED = "openai"
_DIMENSIONS = 512


class ClipImageEmbedder(ImageEmbedder):
    """Generates CLIP image embeddings for visual similarity search."""

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED
        )
        model.eval().to(self._device)
        self._model = model
        self._preprocess = preprocess

    @property
    def dimensions(self) -> int:
        return _DIMENSIONS

    def embed(self, image_bytes: bytes) -> list[float]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().tolist()
