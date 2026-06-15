"""Wire protocol for the pxtfs:// storage service: Pixeltable Cloud home-bucket operations.

Request/response models for obtaining home-bucket credentials and presigned URLs from the Pixeltable Cloud control
plane; see pixeltable.utils.cloud_utils for the client that sends them.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class PixeltableStoreOperationType(str, Enum):
    """Operation types for Pixeltable-managed storage (home buckets)."""

    GET_BUCKET_CREDENTIALS = 'get_bucket_credentials'
    GET_PRESIGNED_URL = 'get_presigned_url'


class GetBucketCredentialsRequest(BaseModel):
    """Request for temporary credentials to access a bucket."""

    operation_type: Literal[PixeltableStoreOperationType.GET_BUCKET_CREDENTIALS] = (
        PixeltableStoreOperationType.GET_BUCKET_CREDENTIALS
    )
    org_slug: str
    db_slug: str
    bucket_name: str = 'home'
    prefix: str | None = None  # Optional path prefix within the home bucket


class GetBucketCredentialsResponse(BaseModel):
    """Response containing temporary credentials for bucket access."""

    access_key_id: str
    secret_access_key: str
    session_token: str
    endpoint_url: str  # S3-compatible endpoint URL for the storage backend
    storage_provider: str = 'r2'  # storage backend type; currently only 'r2' is supported
    resolved_bucket_name: str  # physical bucket name on the storage backend, resolved from the logical bucket name
    ttl_seconds: int  # How long credentials are valid, in seconds
    prefix: str | None = None  # Prefix these credentials are scoped to
    no_space_left: bool = False  # True when storage quota is exceeded; only read and delete are allowed


class GetPresignedUrlRequest(BaseModel):
    """Request for a presigned URL for a key in the org/db home bucket."""

    operation_type: Literal[PixeltableStoreOperationType.GET_PRESIGNED_URL] = (
        PixeltableStoreOperationType.GET_PRESIGNED_URL
    )
    org_slug: str
    db_slug: str
    bucket_name: str = 'home'
    key: str  # Object key within the home bucket
    method: Literal['get', 'put'] = 'get'  # HTTP method for the presigned URL
    expiration: int = 3600  # URL expiry in seconds


class GetPresignedUrlResponse(BaseModel):
    """Response containing a presigned URL for the requested key."""

    url: str
    key: str
    expiration: int
