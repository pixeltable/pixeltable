"""Transport for the Pixeltable cloud management API.

Callers pass a request model from management_protocol and get back the raw response dict.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Literal

import requests

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.service import auth, session_cache
from pixeltable.service.management_protocol import ManagementOperationType
from pixeltable.utils.http import SESSION

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
        ManagementOperationType.LIST_KEYS,
        ManagementOperationType.LIST_DBS,
        ManagementOperationType.GET_DB,
        ManagementOperationType.LIST_SERVICE_INSTANCES,
        ManagementOperationType.GET_SERVICE_INSTANCE,
        ManagementOperationType.GET_LOGS,
    )
)


@dataclasses.dataclass(frozen=True)
class Credential:
    kind: Literal['api_key', 'session']  # API key in header, session token in bearer
    value: str
    source: str

    def header(self) -> dict[str, str]:
        return {'Authorization': f'Bearer {self.value}'} if self.kind == 'session' else {'X-api-key': self.value}


def _api_key_source() -> str:
    env_var = 'PIXELTABLE_API_KEY'
    return (
        f'the {env_var} environment variable' if os.environ.get(env_var) else f'api_key in {Config.get().config_file}'
    )


def configured_credential() -> Credential | None:
    """Return the API key or session credential, depending on what's available."""
    # an API key outranks a session
    api_key = Config.get().get_string_value('api_key')
    if api_key is not None:
        return Credential('api_key', api_key, _api_key_source())
    session = session_cache.load(api_url())
    if session is None:
        return None
    return Credential('session', session.access_token, f'your `pxt login` session for {api_url()}')


def _no_credential(purpose: str) -> excs.Error:
    return excs.AuthorizationError(
        excs.ErrorCode.MISSING_CREDENTIALS,
        f'A Pixeltable API key or sign-in is required to {purpose}. Run `pxt login`, or set an '
        'API key with `os.environ["PIXELTABLE_API_KEY"] = "your-key"` or `api_key = "your-key"` '
        f'in the `[pixeltable]` section of {Config.get().config_file}.\n'
        'For details, see https://docs.pixeltable.com/platform/configuration',
    )


def resolve(purpose: str) -> Credential:
    """The credential to present for `purpose`, renewing a spent session token first."""
    configured = configured_credential()
    if configured is None:
        raise _no_credential(purpose)
    if configured.kind == 'api_key':
        return configured
    token = auth.access_token(api_url())
    if token is None:
        # signed out between the two reads
        raise _no_credential(purpose)
    return dataclasses.replace(configured, value=token)


def raise_if_refused(resp: requests.Response, sent: Credential, operation: str) -> None:
    """Raise for a 401 or a 403, saying which credential the request sent."""
    if resp.status_code not in (401, 403):
        return
    detail = resp.text.strip()
    if resp.status_code == 403:
        # the control plane accepted the credential and refused the operation, so signing in again cannot help
        holder = f'The API key from {sent.source}' if sent.kind == 'api_key' else 'Your Pixeltable session'
        raise excs.AuthorizationError(
            excs.ErrorCode.INSUFFICIENT_PRIVILEGES, f'{holder} is valid but is not permitted to {operation} ({detail}).'
        )
    message = (
        f'The API key from {sent.source} was rejected ({detail}).'
        if sent.kind == 'api_key'
        else f'Your Pixeltable session was rejected ({detail}). {auth.SIGN_IN_AGAIN}'
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
    sent = resolve('reach Pixeltable Cloud')
    headers = {'Content-Type': 'application/json', **sent.header()}
    try:
        resp = SESSION.post(api_url(), data=body, headers=headers, timeout=timeout)
    except requests.exceptions.ConnectionError:
        # a pooled connection closed by the peer while idle fails the call that next picks it up.
        # Retrying gets a new connection, but is only safe for operations that a second delivery
        # cannot change.
        if op_str not in _READ_OPS:
            raise
        resp = SESSION.post(api_url(), data=body, headers=headers, timeout=timeout)
    raise_if_refused(resp, sent, op_str)
    if resp.status_code not in (200, 201):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Management API error {resp.status_code}: {resp.text}',
            provider='pixeltable_cloud',
            status_code=resp.status_code,
        )
    return resp.json()
