"""
Home bucket credential protocol for pixeltable operations.

Defines request/response models for obtaining temporary access credentials
to home buckets (R2-backed blob storage per database).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from .operation_types import PixeltableStoreOperationType


class GetHomeBucketCredentialsRequest(BaseModel):
    """Request for temporary credentials to access a home bucket."""

    operation_type: Literal[PixeltableStoreOperationType.GET_HOME_BUCKET_CREDENTIALS] = (
        PixeltableStoreOperationType.GET_HOME_BUCKET_CREDENTIALS
    )
    org_id: str
    db_slug: str
    prefix: Optional[str] = None  # Optional path prefix within the home bucket


class GetHomeBucketCredentialsResponse(BaseModel):
    """Response containing temporary credentials for home bucket access."""

    access_key_id: str
    secret_access_key: str
    session_token: str
    endpoint_url: str  # S3-compatible endpoint URL for the storage backend
    bucket_name: str  # Resolved R2 bucket name
    ttl_seconds: int  # How long credentials are valid, in seconds
    prefix: Optional[str] = None  # Prefix these credentials are scoped to
    no_space_left: bool = False  # True when storage quota is exceeded; only read and delete are allowed
