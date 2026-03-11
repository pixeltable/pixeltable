"""
Cloud API utilities for pixeltable core.

Provides functions for communicating with the Pixeltable cloud control plane,
such as obtaining temporary credentials for home buckets.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.share.protocol.home_bucket import (
    GetHomeBucketCredentialsRequest,
    GetHomeBucketCredentialsResponse,
)

_logger = logging.getLogger('pixeltable')

PIXELTABLE_API_URL = os.environ.get('PIXELTABLE_API_URL', 'https://internal-api.pixeltable.com')


def _api_headers() -> dict[str, str]:
    headers = {'Content-Type': 'application/json'}
    api_key = Env.get().pxt_api_key
    if api_key is None:
        raise excs.Error(
            'A Pixeltable API key is required for home bucket access. '
            'Set it with `os.environ["PIXELTABLE_API_KEY"] = "your-key"`, '
            f'or add `api_key = "your-key"` to the `[pixeltable]` section in {Config.get().config_file}.\n'
            'For details, see https://docs.pixeltable.com/platform/configuration'
        )
    headers['X-api-key'] = api_key
    return headers


def get_home_bucket_credentials(
    org: str, db: str, prefix: Optional[str] = None
) -> GetHomeBucketCredentialsResponse:
    """
    Fetch temporary R2 credentials for a home bucket from the cloud control plane.

    Args:
        org: Organization slug
        db: Database slug
        prefix: Optional key prefix to scope access within the home bucket

    Returns:
        GetHomeBucketCredentialsResponse with temporary credentials
    """
    request = GetHomeBucketCredentialsRequest(org_slug=org, db_slug=db, prefix=prefix)
    try:
        response = requests.post(
            PIXELTABLE_API_URL, data=request.model_dump_json(), headers=_api_headers(), timeout=15
        )
        if response.status_code != 200:
            raise excs.Error(f'Failed to get home bucket credentials: {response.text}')
        body = response.json()
        if isinstance(body, dict) and 'body' in body:
            import json
            body = json.loads(body['body'])
        return GetHomeBucketCredentialsResponse.model_validate(body)
    except requests.exceptions.RequestException as e:
        raise excs.Error(f'Failed to connect to Pixeltable cloud for home bucket credentials: {e}') from e
