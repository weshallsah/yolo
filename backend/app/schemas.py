from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Base model that (de)serializes as camelCase to match the frontend's types.ts."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class BoundingBox(CamelModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(CamelModel):
    """A single object detected in the image by the YOLO model."""

    label: str
    confidence: float
    box: BoundingBox


class DetectionResponse(CamelModel):
    """API response returned to the frontend for one detection request."""

    detections: list[Detection]
    image_width: int
    image_height: int
