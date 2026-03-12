"""
Presigned URL protocol for pixeltable home bucket operations.

Defines request/response for generating presigned URLs using backend credentials,
so URL expiry is independent of temp credential TTL.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .common import PxtUri, RequestBaseModel
from .operation_types import PixeltableStoreOperationType


class GetPresignedUrlRequest(RequestBaseModel):
    """Request for a presigned URL for a key in the org/db home bucket."""

    operation_type: Literal[PixeltableStoreOperationType.GET_PRESIGNED_URL] = (
        PixeltableStoreOperationType.GET_PRESIGNED_URL
    )
    org_slug: str
    db_slug: str
    key: str  # Object key within the home bucket
    method: Literal['get', 'put'] = 'get'  # HTTP method for the presigned URL
    expiration: int = 3600  # URL expiry in seconds

    def get_pxt_uri(self) -> PxtUri:
        return PxtUri(f'pxt://{self.org_slug}:{self.db_slug}')


class GetPresignedUrlResponse(BaseModel):
    """Response containing a presigned URL for the requested key."""

    url: str
    key: str
    expiration: int
