"""
Cloud API utilities for pixeltable core.

Provides functions for communicating with the Pixeltable cloud management API,
such as obtaining temporary credentials for home buckets.
"""

from __future__ import annotations

from typing import Literal

import requests

from pixeltable import exceptions as excs
from pixeltable.service.management_client import api_url, raise_if_refused, resolve
from pixeltable.service.pxtfs_protocol import (
    GetBucketCredentialsRequest,
    GetBucketCredentialsResponse,
    GetPresignedUrlRequest,
    GetPresignedUrlResponse,
)


def _post(request: GetBucketCredentialsRequest | GetPresignedUrlRequest, timeout: float) -> requests.Response:
    """Send a home-bucket request with an API key if one is set, otherwise the `pxt login` session.

    A refused credential or request raises as it does for a management call, since retrying cannot help.
    """
    purpose = 'reach the home bucket'
    sent = resolve(purpose)
    headers = {'Content-Type': 'application/json', **sent.header()}
    response = requests.post(api_url(), data=request.model_dump_json(), headers=headers, timeout=timeout)
    raise_if_refused(response, sent, purpose)
    return response


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
        response = _post(request, timeout=15)
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
        response = _post(request, timeout=30)
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
