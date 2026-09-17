from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class BoundingBox(CamelModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(CamelModel):
    label: str
    confidence: float
    box: BoundingBox


class DetectionResponse(CamelModel):
    detections: list[Detection]
    image_width: int
    image_height: int
