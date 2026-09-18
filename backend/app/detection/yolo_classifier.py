import io

from PIL import Image
from ultralytics import YOLO

from app.detection.base import ObjectDetector
from app.schemas import BoundingBox, Detection


class YoloImageClassifier(ObjectDetector):
    """Serves a YOLOv8 classification checkpoint through the detection interface.

    A classifier scores the whole image and produces no boxes, so every prediction is
    reported over the full frame. That is the honest extent of what it localizes: the label
    describes the image, not any region inside it. Callers that need real boxes need a
    detector trained on real boxes, which this dataset does not provide.
    """

    def __init__(self, weights_path: str, confidence_threshold: float, top_k: int = 5) -> None:
        self._model = YOLO(weights_path)
        self._confidence_threshold = confidence_threshold
        self._top_k = top_k

    def detect(self, image_bytes: bytes) -> list[Detection]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = self._model.predict(image, verbose=False)

        full_frame = BoundingBox(x1=0.0, y1=0.0, x2=float(image.width), y2=float(image.height))
        detections = [
            Detection(
                label=result.names[int(index)],
                confidence=float(result.probs.data[int(index)]),
                box=full_frame,
            )
            for result in results
            for index in result.probs.top5[: self._top_k]
            if float(result.probs.data[int(index)]) >= self._confidence_threshold
        ]

        return sorted(detections, key=lambda detection: detection.confidence, reverse=True)
