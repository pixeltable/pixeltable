"""Transport for the Pixeltable cloud management API.

Callers pass a request model from management_protocol and get back the raw response dict.
"""

from __future__ import annotations

import http.cookiejar
import os
from typing import Any

import requests
from requests.adapters import HTTPAdapter, Retry

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.service import auth, credentials
from pixeltable.service.management_protocol import ManagementOperationType

_DEFAULT_API_URL = 'https://internal-api.pixeltable.com'


def api_url() -> str:
    """URL of the Pixeltable cloud management API."""
    # api_url is deliberately not a registered config option: PIXELTABLE_API_URL is the only override
    url = Config.get().get_string_value('api_url')
    return _DEFAULT_API_URL if url is None else url


_LONG_OPS = frozenset(
    op.value
    for op in (
        ManagementOperationType.CREATE_DB,
        ManagementOperationType.BUILD_IMAGE,
        ManagementOperationType.DELETE_DB,
        ManagementOperationType.GET_LOGS,
    )
)

# operations that don't change server state; can be sent multiple times
_READ_OPS = frozenset(
    op.value
    for op in (
        ManagementOperationType.LIST_ORGS,
        ManagementOperationType.LIST_DBS,
        ManagementOperationType.GET_DB,
        ManagementOperationType.LIST_SERVICE_INSTANCES,
        ManagementOperationType.GET_SERVICE_INSTANCE,
        ManagementOperationType.GET_LOGS,
    )
)

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


# Whatever went wrong with a session, the fix is the same and there is nothing to decide between,
# so it is one line rather than a diagnosis.
_SESSION_FAILED = 'Your Pixeltable session may have expired. Run `pxt login` again, or set an API key.'


def credential(purpose: str) -> str:
    """The credential to present for `purpose`: an API key if one is set, else a `pxt login` session.

    The key wins when both are present. It is the explicit choice -- set in the environment or the
    config file, and what CI uses -- so an ambient browser session must not quietly take precedence
    over it.

    Returned as a bare string because every consumer sends it differently -- a header for the
    control plane, a CONNECT frame for the tunnel -- and because the sidecar tells the two apart by
    shape (a JWT has two dots; a WorkOS API key never does), so neither needs to be told which it got.
    """
    api_key = Env.get().pxt_api_key
    if api_key is not None:
        return api_key
    try:
        token = auth.access_token(api_url())
    except auth.AuthError as e:
        # The underlying reason stays on the exception chain; it is not something the user acts on.
        raise excs.AuthorizationError(excs.ErrorCode.MISSING_CREDENTIALS, _SESSION_FAILED) from e
    if token is None:
        # The one message that spells the options out: nothing is set up yet, so the config file and
        # the docs are what the reader actually needs here.
        raise excs.AuthorizationError(
            excs.ErrorCode.MISSING_CREDENTIALS,
            f'A Pixeltable API key or sign-in is required to {purpose}. Run `pxt login`, or set an '
            'API key with `os.environ["PIXELTABLE_API_KEY"] = "your-key"` or `api_key = "your-key"` '
            f'in the `[pixeltable]` section of {Config.get().config_file}.\n'
            'For details, see https://docs.pixeltable.com/platform/configuration',
        )
    return token


def credential_header(purpose: str) -> dict[str, str]:
    """How to send that credential. One place decides, so no caller has to know which kind it got."""
    cred = credential(purpose)
    return {'Authorization': f'Bearer {cred}'} if cred.count('.') == 2 else {'X-api-key': cred}


def _api_headers() -> dict[str, str]:
    return {'Content-Type': 'application/json', **credential_header('reach Pixeltable Cloud')}


def credential_source() -> tuple[str, str]:
    """Which credential this client will send, and where it came from.

    Reported rather than inferred because the two are easy to confuse: an API key set once in the
    config file silently outranks a session created seconds ago, and the resulting failure names
    neither.
    """
    if os.environ.get('PIXELTABLE_API_KEY'):
        return 'api_key', 'the PIXELTABLE_API_KEY environment variable'
    if Env.get().pxt_api_key is not None:
        return 'api_key', f'api_key in {Config.get().config_file}'
    if credentials.load(api_url()) is not None:
        return 'session', f'your `pxt login` session for {api_url()}'
    return 'none', 'nothing'


def _raise_unauthorized(resp: Any) -> None:
    """Turn a 401 into an error that names the credential that was actually sent.

    A bare "unauthorized" does not say which of the two was tried, which is the whole question when
    a machine has both.
    """
    kind, where = credential_source()
    detail = resp.text.strip()
    if kind == 'api_key':
        # No suggestion to sign in instead: a key always outranks a session, so signing in would
        # change nothing. The key is what needs fixing.
        raise excs.AuthorizationError(
            excs.ErrorCode.MISSING_CREDENTIALS, f'The API key from {where} was rejected ({detail}).'
        )
    raise excs.AuthorizationError(excs.ErrorCode.MISSING_CREDENTIALS, _SESSION_FAILED)


def api_call(request: Any) -> dict[str, Any]:
    """Forward one request to the cloud management API and return the raw response dict."""
    op = getattr(request, 'operation_type', None)
    op_str = op.value if hasattr(op, 'value') else str(op) if op else ''
    timeout = 180 if op_str in _LONG_OPS else 30
    # by_alias: a field the control plane names differently declares that name as its alias
    body = request.model_dump_json(by_alias=True)
    try:
        resp = _SESSION.post(api_url(), data=body, headers=_api_headers(), timeout=timeout)
    except requests.exceptions.ConnectionError:
        # a pooled connection closed by the peer while idle fails the call that next picks it up.
        # Retrying gets a new connection, but is only safe for operations that a second delivery
        # cannot change.
        if op_str not in _READ_OPS:
            raise
        resp = _SESSION.post(api_url(), data=body, headers=_api_headers(), timeout=timeout)
    if resp.status_code == 401:
        _raise_unauthorized(resp)
    if resp.status_code not in (200, 201):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Management API error {resp.status_code}: {resp.text}',
            provider='pixeltable_cloud',
            status_code=resp.status_code,
        )
    return resp.json()
