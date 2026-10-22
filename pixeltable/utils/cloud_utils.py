"""
Cloud API utilities for pixeltable core.

Provides functions for communicating with the Pixeltable cloud management API,
such as obtaining temporary credentials for home buckets.
"""

from __future__ import annotations

from typing import Literal

import requests
from pydantic import BaseModel

from pixeltable import exceptions as excs
from pixeltable.service.management_client import api_url, raise_if_refused, resolve
from pixeltable.service.pxtfs_protocol import (
    GetBucketCredentialsRequest,
    GetBucketCredentialsResponse,
    GetPresignedUrlRequest,
    GetPresignedUrlResponse,
)
from pixeltable.utils.http import SESSION


class _GetPresignedUrlsRequest(BaseModel):
    """Request for GET URLs for keys in the org/db home bucket. The control plane defines this operation."""

    operation_type: Literal['get_presigned_urls'] = 'get_presigned_urls'
    org: str
    db: str
    bucket_name: str = 'home'
    keys: list[str]
    expires_in: int  # URL expiry in seconds


class _GetPresignedUrlsResponse(BaseModel):
    urls: dict[str, str]  # by key
    expires_in: int


def _post(
    request: GetBucketCredentialsRequest | GetPresignedUrlRequest | _GetPresignedUrlsRequest,
) -> requests.Response:
    """Send a home-bucket request with an API key if one is set, otherwise the `pxt login` session.

    A refused credential or request raises as it does for a management call, since retrying cannot help.
    """
    purpose = 'reach the home bucket'
    sent = resolve(purpose)
    headers = {'Content-Type': 'application/json', **sent.header()}
    body = request.model_dump_json()
    # a control plane starting cold takes 17-19 s to answer, outside prod's provisioned instances
    timeout = 30
    try:
        response = SESSION.post(api_url(), data=body, headers=headers, timeout=timeout)
    except requests.exceptions.ConnectionError:
        # a pooled connection closed by the peer while idle fails the call that next picks it up; these
        # requests only read, so sending one again on a new connection is safe
        response = SESSION.post(api_url(), data=body, headers=headers, timeout=timeout)
    raise_if_refused(response, sent, purpose)
    return response


def _cloud_error(failure: str, cause: object, status_code: int | None = None) -> excs.ExternalServiceError:
    """The error for a request to Pixeltable Cloud that failed: '<failure>: <cause>'."""
    return excs.ExternalServiceError(
        excs.ErrorCode.PROVIDER_ERROR, f'{failure}: {cause}', provider='pixeltable_cloud', status_code=status_code
    )


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
        response = _post(request)
        if response.status_code != 200:
            raise _cloud_error('Failed to get bucket credentials', response.text, response.status_code)
        data = response.json()
        return GetBucketCredentialsResponse.model_validate(data)
    except requests.exceptions.RequestException as e:
        raise _cloud_error('Failed to connect to Pixeltable Cloud for bucket credentials', e) from e


def get_presigned_url_from_cloud(
    org: str, db: str, bucket: str, key: str, method: Literal['get', 'put'] = 'get', expiration: int = 3600
) -> str:
    """
    Request a presigned URL from Pixeltable Cloud for a key in given bucket.
    Uses backend credentials on the cloud so URL expiry is independent of temp credential TTL.
    """
    request = GetPresignedUrlRequest(org=org, db=db, bucket_name=bucket, key=key, method=method, expiration=expiration)
    failure = 'Failed to get presigned URL from Pixeltable Cloud'
    try:
        response = _post(request)
        if response.status_code != 200:
            raise _cloud_error(failure, response.text, response.status_code)
        data = response.json()
        return GetPresignedUrlResponse.model_validate(data).url
    except requests.exceptions.RequestException as e:
        raise _cloud_error(failure, e) from e


MAX_PRESIGNED_URL_KEYS = 100  # keys per get_presigned_urls call: the control plane's limit


def get_presigned_urls_from_cloud(org: str, db: str, bucket: str, keys: list[str], expires_in: int) -> dict[str, str]:
    """
    Request presigned GET URLs from Pixeltable Cloud for keys in the given bucket of org/db, in one call.
    The control plane signs at most MAX_PRESIGNED_URL_KEYS keys a call. Returns each key's URL.
    """
    request = _GetPresignedUrlsRequest(org=org, db=db, bucket_name=bucket, keys=keys, expires_in=expires_in)
    failure = 'Failed to get presigned URLs from Pixeltable Cloud'
    try:
        response = _post(request)
    except requests.exceptions.RequestException as e:
        raise _cloud_error(failure, e) from e
    if response.status_code != 200:
        raise _cloud_error(failure, response.text, response.status_code)
    try:
        urls = _GetPresignedUrlsResponse.model_validate(response.json()).urls
    except ValueError:
        # a JSON or validation error quotes the answer, whose URLs are credentials
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            'Pixeltable Cloud returned a malformed answer to get_presigned_urls',
            provider='pixeltable_cloud',
        ) from None
    missing = [key for key in keys if key not in urls]
    if len(missing) > 0:
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Pixeltable Cloud returned no presigned URL for {len(missing)} of {len(keys)} keys, '
            f'including {missing[0]!r}',
            provider='pixeltable_cloud',
        )
    return urls
