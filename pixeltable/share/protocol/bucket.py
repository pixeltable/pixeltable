"""
Defines request/response models for obtaining temporary access credentials for cloud registered buckets
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from .common import PxtUri, RequestBaseModel
from .operation_types import PixeltableStoreOperationType


class GetBucketCredentialsRequest(RequestBaseModel):
    """Request for temporary credentials to access a bucket."""

    operation_type: Literal[PixeltableStoreOperationType.GET_BUCKET_CREDENTIALS] = (
        PixeltableStoreOperationType.GET_BUCKET_CREDENTIALS
    )
    org_slug: str
    db_slug: str
    bucket_name: str = 'home'
    prefix: str | None = None  # Optional path prefix within the home bucket

    def get_pxt_uri(self) -> PxtUri:
        return PxtUri(f'pxt://{self.org_slug}:{self.db_slug}/{self.bucket_name}')


class GetBucketCredentialsResponse(BaseModel):
    """Response containing temporary credentials for bucket access."""

    access_key_id: str
    secret_access_key: str
    session_token: str
    endpoint_url: str  # S3-compatible endpoint URL for the storage backend
    storage_provider: str = 'r2'  # storage backend type; currently only 'r2' is supported
    resolved_bucket_name: str  # physical bucket name on the storage backend, resolved from the logical bucket name
    ttl_seconds: int  # How long credentials are valid, in seconds
    prefix: Optional[str] = None  # Prefix these credentials are scoped to
    no_space_left: bool = False  # True when storage quota is exceeded; only read and delete are allowed
