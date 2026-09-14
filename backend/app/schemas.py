from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

TrustLevel = Literal["verified", "caution", "unverified"]


class CamelModel(BaseModel):
    """Base model that (de)serializes as camelCase to match the frontend's types.ts."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Detection(BaseModel):
    """Raw output of an object detector for a single detected object."""

    label: str
    confidence: float


class PriceListing(CamelModel):
    source: str
    price: float
    currency: str
    condition: str
    url: str
    thumbnail: str | None = None
    is_best_value: bool = False


class Trust(BaseModel):
    level: TrustLevel
    reason: str


class IdentifyResult(CamelModel):
    """API response returned to the frontend for one identified product."""

    id: str
    title: str
    category: str
    description: str
    confidence: float
    trust: Trust
    price_listings: list[PriceListing]
    average_price: float
    recommendation: str
