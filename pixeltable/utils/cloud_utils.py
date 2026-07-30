"""
Cloud API utilities for pixeltable core.

Provides functions for communicating with the Pixeltable cloud management API,
such as obtaining temporary credentials for home buckets.
"""

from __future__ import annotations

from typing import Literal

import requests

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.service.management_client import api_url
from pixeltable.service.pxtfs_protocol import (
    GetBucketCredentialsRequest,
    GetBucketCredentialsResponse,
    GetPresignedUrlRequest,
    GetPresignedUrlResponse,
)


def _api_headers() -> dict[str, str]:
    headers = {'Content-Type': 'application/json'}
    api_key = Env.get().pxt_api_key
    if api_key is None:
        raise excs.AuthorizationError(
            excs.ErrorCode.MISSING_CREDENTIALS,
            'A Pixeltable API key is required for home bucket access. '
            'Set it with `os.environ["PIXELTABLE_API_KEY"] = "your-key"`, '
            f'or add `api_key = "your-key"` to the `[pixeltable]` section in {Config.get().config_file}.\n'
            'For details, see https://docs.pixeltable.com/platform/configuration',
        )
    headers['X-api-key'] = api_key
    return headers


def get_bucket_credentials(org: str, db: str, bucket: str, prefix: str | None = None) -> GetBucketCredentialsResponse:
    """
    Fetch temporary R2 credentials for a home bucket from the cloud management API.

    Args:
        org: Organization name
        db: Database name
        bucket: Bucket name registered
        prefix: Optional key prefix to scope access within the home bucket

    Returns:
        GetBucketCredentialsResponse with temporary credentials
    """
    request = GetBucketCredentialsRequest(org=org, db=db, bucket_name=bucket, prefix=prefix)
    try:
        response = requests.post(api_url(), data=request.model_dump_json(), headers=_api_headers(), timeout=15)
        if response.status_code != 200:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'Failed to get bucket credentials: {response.text}',
                provider='pixeltable_cloud',
                status_code=response.status_code,
            )
        data = response.json()
        return GetBucketCredentialsResponse.model_validate(data)
    except requests.exceptions.RequestException as e:
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Failed to connect to Pixeltable Cloud for bucket credentials: {e}',
            provider='pixeltable_cloud',
        ) from e


def get_presigned_url_from_cloud(
    org: str, db: str, bucket: str, key: str, method: Literal['get', 'put'] = 'get', expiration: int = 3600
) -> str:
    """
    Request a presigned URL from Pixeltable Cloud for a key in given bucket.
    Uses backend credentials on the cloud so URL expiry is independent of temp credential TTL.
    """
    request = GetPresignedUrlRequest(org=org, db=db, bucket_name=bucket, key=key, method=method, expiration=expiration)
    try:
        response = requests.post(api_url(), data=request.model_dump_json(), headers=_api_headers(), timeout=30)
        if response.status_code != 200:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'Failed to get presigned URL from Pixeltable Cloud: {response.text}',
                provider='pixeltable_cloud',
                status_code=response.status_code,
            )
        data = response.json()
        return GetPresignedUrlResponse.model_validate(data).url
    except requests.exceptions.RequestException as e:
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Failed to get presigned URL from Pixeltable Cloud: {e}',
            provider='pixeltable_cloud',
        ) from e
