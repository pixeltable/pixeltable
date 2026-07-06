"""Protocol models for Pixeltable home bucket credential and presigned URL operations."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from .operation_types import PixeltableStoreOperationType


class GetBucketCredentialsRequest(BaseModel):
    operation_type: Literal[PixeltableStoreOperationType.GET_BUCKET_CREDENTIALS] = (
        PixeltableStoreOperationType.GET_BUCKET_CREDENTIALS
    )
    org_slug: str
    db_slug: str
    bucket_name: str
    prefix: Optional[str] = None


class GetBucketCredentialsResponse(BaseModel):
    access_key_id: str
    secret_access_key: str
    session_token: str
    endpoint_url: str
    storage_provider: str
    resolved_bucket_name: str
    ttl_seconds: int
    prefix: Optional[str] = None
    no_space_left: bool = False


class GetPresignedUrlRequest(BaseModel):
    operation_type: Literal[PixeltableStoreOperationType.GET_PRESIGNED_URL] = (
        PixeltableStoreOperationType.GET_PRESIGNED_URL
    )
    org_slug: str
    db_slug: str
    bucket_name: str
    key: str
    method: str = 'get'
    expiration: int = 3600


class GetPresignedUrlResponse(BaseModel):
    url: str
    key: str
    expiration: int
