import io

from PIL import Image
from ultralytics import YOLO

from app.detection.base import ObjectDetector
from app.schemas import BoundingBox, Detection


class YoloObjectDetector(ObjectDetector):
    def __init__(self, weights_path: str, confidence_threshold: float) -> None:
        self._model = YOLO(weights_path)
        self._confidence_threshold = confidence_threshold

    def detect(self, image_bytes: bytes) -> list[Detection]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = self._model.predict(image, verbose=False)

        detections = [
            Detection(
                label=result.names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                box=BoundingBox(
                    x1=float(box.xyxy[0][0]),
                    y1=float(box.xyxy[0][1]),
                    x2=float(box.xyxy[0][2]),
                    y2=float(box.xyxy[0][3]),
                ),
            )
            for result in results
            for box in result.boxes
            if float(box.conf[0]) >= self._confidence_threshold
        ]

        return sorted(detections, key=lambda detection: detection.confidence, reverse=True)
