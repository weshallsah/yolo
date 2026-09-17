import io

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from PIL import Image

from app.dependencies import get_detector
from app.detection.base import ObjectDetector
from app.schemas import DetectionResponse

router = APIRouter(prefix="/api", tags=["detect"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def _read_image_bytes(image: UploadFile) -> bytes:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Please upload a JPG, PNG, or WEBP image.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    return image_bytes


@router.post("/detect", response_model=DetectionResponse)
async def detect(
    image: UploadFile,
    detector: ObjectDetector = Depends(get_detector),
) -> DetectionResponse:
    image_bytes = await _read_image_bytes(image)

    try:
        width, height = Image.open(io.BytesIO(image_bytes)).size
    except Exception as error:
        raise HTTPException(status_code=400, detail="Could not read this image.") from error

    detections = detector.detect(image_bytes)

    return DetectionResponse(detections=detections, image_width=width, image_height=height)
