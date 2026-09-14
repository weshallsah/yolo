from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from app.dependencies import get_identify_service
from app.schemas import IdentifyResult
from app.services.exceptions import IdentificationError
from app.services.identify_service import IdentifyService

router = APIRouter(prefix="/api", tags=["identify"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def _read_image_bytes(image: UploadFile) -> bytes:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Please upload a JPG, PNG, or WEBP image.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    return image_bytes


@router.post("/identify", response_model=IdentifyResult)
async def identify(
    image: UploadFile,
    service: IdentifyService = Depends(get_identify_service),
) -> IdentifyResult:
    image_bytes = await _read_image_bytes(image)

    try:
        return service.identify(image_bytes)
    except IdentificationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/identify/manual", response_model=IdentifyResult)
async def identify_manual(
    image: UploadFile,
    label: str = Form(...),
    service: IdentifyService = Depends(get_identify_service),
) -> IdentifyResult:
    image_bytes = await _read_image_bytes(image)

    if not label.strip():
        raise HTTPException(status_code=400, detail="Please tell us what this item is.")

    try:
        return service.identify_manual(image_bytes, label.strip())
    except IdentificationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
