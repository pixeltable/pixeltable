"""Transport for the Pixeltable cloud management API.

Callers pass a request model from management_protocol and get back the raw response dict.
"""

from __future__ import annotations

import http.cookiejar
from typing import Any

import requests
from requests.adapters import HTTPAdapter, Retry

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env

_DEFAULT_API_URL = 'https://internal-api.pixeltable.com'


def api_url() -> str:
    """URL of the Pixeltable cloud management API."""
    # api_url is deliberately not a registered config option: PIXELTABLE_API_URL is the only override
    url = Config.get().get_string_value('api_url')
    return _DEFAULT_API_URL if url is None else url


_LONG_OPS = frozenset({'create_db', 'update_runtime', 'delete_db'})

# operations that don't change server state; can be sent multiple times
_READ_OPS = frozenset({'list_orgs', 'list_dbs', 'get_db', 'list_services', 'get_service'})

# maximum number of connections kept open to the service, sized for concurrent calls from multiple threads
_POOL_MAXSIZE = 16


def _new_session() -> requests.Session:
    """Create a pooled session for management API requests."""
    session = requests.Session()
    # the management API sets no cookies; blocking the jar leaves the session without mutable state, so
    # concurrent calls can share it
    session.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
    # retry only failures to establish a connection: those never reached the server. allowed_methods
    # gates read and status retries alone, so an empty set still leaves connect retries on, while
    # keeping a POST from being replayed after the server may have already processed it.
    retries = Retry(total=2, connect=2, read=0, status=0, other=0, allowed_methods=frozenset(), backoff_factor=0.2)
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=_POOL_MAXSIZE, max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


_SESSION = _new_session()


def _api_headers() -> dict[str, str]:
    return {'Content-Type': 'application/json', 'X-api-key': Env.get().require_api_key()}


def api_call(request: Any) -> dict[str, Any]:
    """Forward one request to the cloud management API and return the raw response dict."""
    op = getattr(request, 'operation_type', None)
    op_str = op.value if hasattr(op, 'value') else str(op) if op else ''
    timeout = 180 if op_str in _LONG_OPS else 30
    body = request.model_dump_json()
    try:
        resp = _SESSION.post(api_url(), data=body, headers=_api_headers(), timeout=timeout)
    except requests.exceptions.ConnectionError:
        # a pooled connection closed by the peer while idle fails the call that next picks it up.
        # Retrying gets a new connection, but is only safe for operations that a second delivery
        # cannot change.
        if op_str not in _READ_OPS:
            raise
        resp = _SESSION.post(api_url(), data=body, headers=_api_headers(), timeout=timeout)
    if resp.status_code not in (200, 201):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Management API error {resp.status_code}: {resp.text}',
            provider='pixeltable_cloud',
            status_code=resp.status_code,
        )
    return resp.json()
