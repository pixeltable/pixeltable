"""Transport for the Pixeltable cloud management API.

Callers pass a request model from management_protocol and get back the raw response dict.
"""

from __future__ import annotations

import http.cookiejar
import os
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from pixeltable import exceptions as excs
from pixeltable.config import Config
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
    for op in (ManagementOperationType.UPDATE_DB, ManagementOperationType.DELETE_DB, ManagementOperationType.GET_LOGS)
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


_SIGN_IN_AGAIN = 'Run `pxt login` again, or set an API key.'


def _api_key() -> Optional[str]:
    """The configured API key, read without standing up an Env.

    Env.get() initialises the local database, which signing in has no use for: `pxt login` should
    not start postgres to read a string out of the config file.
    """
    return Config.get().get_string_value('api_key')


def credential(purpose: str) -> str:
    """The credential to present for `purpose`: an API key if one is set, else a `pxt login` session.

    A key outranks a session: setting one is the explicit choice, and what CI runs on. The string is
    returned bare because consumers send it differently -- a header here, a CONNECT frame in the
    tunnel -- and because both ends tell the two apart by shape.
    """
    api_key = _api_key()
    if api_key is not None:
        return api_key
    try:
        token = auth.access_token(api_url())
    except auth.AuthError as e:
        # Whether the sign-in is over is knowable here, from the cache and from what WorkOS said.
        # Everything else -- a name that will not resolve, a provider having a bad day -- is reported
        # as itself: telling someone to sign in again does not fix a network they cannot reach.
        if e.code == auth.SESSION_EXPIRED:
            detail = f'Your Pixeltable sign-in has expired. {_SIGN_IN_AGAIN}'
        elif e.code in auth.NEEDS_SIGN_IN:
            detail = f'Your Pixeltable session was rejected ({e}). {_SIGN_IN_AGAIN}'
        else:
            detail = f'Could not use your Pixeltable session: {e}.'
        raise excs.AuthorizationError(excs.ErrorCode.MISSING_CREDENTIALS, detail) from e
    if token is None:
        # Nothing is set up yet, so this is the one message that spells both ways out in full.
        raise excs.AuthorizationError(
            excs.ErrorCode.MISSING_CREDENTIALS,
            f'A Pixeltable API key or sign-in is required to {purpose}. Run `pxt login`, or set an '
            'API key with `os.environ["PIXELTABLE_API_KEY"] = "your-key"` or `api_key = "your-key"` '
            f'in the `[pixeltable]` section of {Config.get().config_file}.\n'
            'For details, see https://docs.pixeltable.com/platform/configuration',
        )
    return token


def credential_header(purpose: str) -> dict[str, str]:
    """That credential as a header. A JWT has two dots; a WorkOS API key never does."""
    cred = credential(purpose)
    return {'Authorization': f'Bearer {cred}'} if cred.count('.') == 2 else {'X-api-key': cred}


def _api_headers() -> dict[str, str]:
    return {'Content-Type': 'application/json', **credential_header('reach Pixeltable Cloud')}


def credential_source() -> tuple[str, str]:
    """Which credential this client will send, and where it came from. For `pxt whoami` and errors."""
    if os.environ.get('PIXELTABLE_API_KEY'):
        return 'api_key', 'the PIXELTABLE_API_KEY environment variable'
    if _api_key() is not None:
        return 'api_key', f'api_key in {Config.get().config_file}'
    if credentials.load(api_url()) is not None:
        return 'session', f'your `pxt login` session for {api_url()}'
    return 'none', 'nothing'


def _raise_unauthorized(resp: Any) -> None:
    """Report a 401 naming the credential that was sent, so the reader knows where to look."""
    kind, where = credential_source()
    detail = resp.text.strip()
    message = (
        f'The API key from {where} was rejected ({detail}).'
        if kind == 'api_key'
        else f'Your Pixeltable session was rejected ({detail}). {_SIGN_IN_AGAIN}'
    )
    # PROVIDER_AUTH_ERROR, not PROVIDER_ERROR: a refused credential is not retryable, and retrying
    # one only delays the error. A 401 is always the control plane's own decision -- it answers 503,
    # never 401, when WorkOS is the thing that could not be reached.
    raise excs.ExternalServiceError(
        excs.ErrorCode.PROVIDER_AUTH_ERROR, message, provider='pixeltable_cloud', status_code=resp.status_code
    )


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
