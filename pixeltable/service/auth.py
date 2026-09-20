"""Signing in from the CLI, and keeping that session alive.

`pxt login` uses the OAuth device authorization grant (RFC 8628): the CLI asks for a code, you
approve it in a browser, and the CLI polls until you have. Nothing secret travels through a URL, and
the CLI keeps its own refresh token rather than a copy of the browser's session, so renewing on one
side cannot invalidate the other. Renewal is silent for as long as the refresh token is honored.

The CLI is a public client, so no request here sends a client secret.

Every endpoint comes from the issuer's OIDC metadata, and the control plane points at the issuer, so
no sign-in URL is compiled in here.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import json
import threading
import time
import urllib.parse
from typing import Any

import requests

from pixeltable import exceptions as excs
from pixeltable.service import session_cache
from pixeltable.service.session_cache import Session
from pixeltable.utils.http import SESSION
from pixeltable_cli.models import LoginStartResponse

# Where a control plane says which issuer signs people in to it; served without authentication.
_AUTH_CONFIG_PATH = '/.well-known/pixeltable-auth'
# The issuer's own metadata: OIDC Discovery 1.0, registered by RFC 8414.
_OIDC_CONFIG_PATH = '/.well-known/openid-configuration'

# What an error calls it: the user never chose WorkOS and cannot act on its name.
_SIGN_IN_SERVICE = 'the Pixeltable sign-in service'
_DEVICE_GRANT = 'urn:ietf:params:oauth:grant-type:device_code'

# Ending the browser's own sign-in has no metadata field to discover it, so WorkOS's URL stands.
_WORKOS_LOGOUT = 'https://api.workos.com/user_management/sessions/logout'

# WorkOS's code for a refresh token it no longer honors.
_REJECTED_GRANT = 'invalid_grant'

SIGN_IN_AGAIN = 'Run `pxt login` again, or set an API key.'

_TIMEOUT_S = 30.0


@dataclasses.dataclass(frozen=True)
class TokenErrorResponse:
    """The token endpoint's error response, RFC 6749 section 5.2."""

    code: str
    description: str


def _unreachable(what: str, exc: Exception) -> excs.Error:
    return excs.ExternalServiceError(
        excs.ErrorCode.PROVIDER_ERROR, f'Could not reach {what}: {exc}', provider='pixeltable_cloud'
    )


def _bad_status(what: str, resp: requests.Response) -> excs.Error:
    return excs.ExternalServiceError(
        excs.ErrorCode.PROVIDER_ERROR,
        f'{what} returned HTTP {resp.status_code}.',
        provider='pixeltable_cloud',
        status_code=resp.status_code,
    )


def _payload(what: str, resp: requests.Response) -> dict[str, Any]:
    """The JSON object in a response. Anything else raises InternalError: the two ends disagree on the protocol."""
    try:
        body = resp.json()
    except ValueError as e:
        raise excs.InternalError(excs.ErrorCode.INTERNAL_ERROR, f'{what} did not answer with JSON') from e
    if not isinstance(body, dict):
        raise excs.InternalError(
            excs.ErrorCode.INTERNAL_ERROR, f'{what} answered with {type(body).__name__}, not an object'
        )
    return body


def _json_request(url: str, what: str, fields: dict[str, str] | None = None) -> dict[str, Any]:
    """GET url, or POST fields to it as a form, and return the JSON object in the answer.

    Anything but 200 or 201 raises.
    """
    try:
        resp = (
            SESSION.get(url, timeout=_TIMEOUT_S)
            if fields is None
            else SESSION.post(url, data=fields, timeout=_TIMEOUT_S)
        )
    except requests.RequestException as e:
        raise _unreachable(what, e) from e
    if resp.status_code not in (200, 201):
        raise _bad_status(what, resp)
    return _payload(what, resp)


def _token_request(api_url: str, fields: dict[str, str]) -> dict[str, Any] | TokenErrorResponse:
    """The token fields, or the OAuth error code and description.

    A 4xx with an OAuth error body is returned rather than raised: under the device grant it reports
    a pending approval or too fast a poll, neither of which is a failure.
    """
    try:
        resp = SESSION.post(sign_in_config(api_url).token_endpoint, data=fields, timeout=_TIMEOUT_S)
    except requests.RequestException as e:
        raise _unreachable(_SIGN_IN_SERVICE, e) from e
    if resp.status_code in (200, 201):
        return _payload(_SIGN_IN_SERVICE, resp)
    try:
        body = resp.json()
    except ValueError:
        raise _bad_status(_SIGN_IN_SERVICE, resp) from None
    code = str(body.get('error') or '') if isinstance(body, dict) else ''
    if code == '':
        raise _bad_status(_SIGN_IN_SERVICE, resp)
    return TokenErrorResponse(code, str(body.get('error_description') or ''))


@dataclasses.dataclass(frozen=True)
class SignInConfig:
    """One control plane's sign-in endpoints, and the public client to present at them."""

    client_id: str
    device_authorization_endpoint: str
    token_endpoint: str


_config_cache: dict[str, SignInConfig] = {}  # key: API URL
_config_lock = threading.Lock()


def sign_in_config(api_url: str) -> SignInConfig:
    """Read this control plane's public client and its issuer's grant endpoints into _config_cache."""
    with _config_lock:
        cached = _config_cache.get(api_url)
        if cached is not None:
            return cached
        config = _json_request(api_url.rstrip('/') + _AUTH_CONFIG_PATH, api_url)
        client_id = str(config.get('client_id') or '')
        issuer = str(config.get('issuer') or '').rstrip('/')
        if client_id == '' or issuer == '':
            raise excs.InternalError(
                excs.ErrorCode.INTERNAL_ERROR, f'{api_url} did not say which sign-in service to use'
            )

        metadata = _json_request(issuer + _OIDC_CONFIG_PATH, _SIGN_IN_SERVICE)
        device_authorization_endpoint = str(metadata.get('device_authorization_endpoint') or '')
        token_endpoint = str(metadata.get('token_endpoint') or '')
        missing = [
            name
            for name, value in (
                ('device_authorization_endpoint', device_authorization_endpoint),
                ('token_endpoint', token_endpoint),
            )
            if value == ''
        ]
        if len(missing) > 0:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'{issuer} cannot sign in a CLI: its OIDC metadata omits {", ".join(missing)}.',
                provider='pixeltable_cloud',
            )
        resolved = SignInConfig(client_id, device_authorization_endpoint, token_endpoint)
        _config_cache[api_url] = resolved
        return resolved


def _claims(token: str) -> dict[str, Any]:
    """A token's payload. Read, never verified: nothing here is an authorization decision."""
    try:
        payload = token.split('.')[1]
        decoded = json.loads(base64.urlsafe_b64decode(payload + '=' * (-len(payload) % 4)))
    except (IndexError, ValueError, binascii.Error):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _number(payload: dict[str, Any], key: str, default: float) -> float:
    """A numeric field, or the default when it is absent or unusable. `or` would reject a real 0."""
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def _create_session(payload: dict[str, Any], client_id: str) -> Session:
    token = str(payload.get('access_token') or '')
    if token == '':
        raise excs.InternalError(excs.ErrorCode.INTERNAL_ERROR, f'{_SIGN_IN_SERVICE} returned no access token')
    user = payload.get('user') or {}
    exp = _claims(token).get('exp')
    expires_at = float(exp) if isinstance(exp, (int, float)) else time.time() + 300.0
    return Session(
        access_token=token,
        expires_at=expires_at,
        refresh_token=str(payload.get('refresh_token') or ''),
        client_id=client_id,
        email=str(user.get('email') or ''),
        organization_id=str(payload.get('organization_id') or ''),
    )


def device_login_start(api_url: str) -> LoginStartResponse:
    """Start the device flow. This returns as soon as the code exists: no browser, no waiting."""
    resolved = sign_in_config(api_url)
    client_id = resolved.client_id
    authz_resp = _json_request(resolved.device_authorization_endpoint, _SIGN_IN_SERVICE, {'client_id': client_id})
    device_code = str(authz_resp.get('device_code') or '')
    user_code = str(authz_resp.get('user_code') or '')
    verification_uri = str(authz_resp.get('verification_uri_complete') or authz_resp.get('verification_uri') or '')
    if device_code == '' or user_code == '' or verification_uri == '':
        raise excs.InternalError(excs.ErrorCode.INTERNAL_ERROR, f'{_SIGN_IN_SERVICE} did not return a device code')
    return LoginStartResponse(
        client_id=client_id,
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        interval=_number(authz_resp, 'interval', 5.0),
        expires_in=_number(authz_resp, 'expires_in', 300.0),
    )


def device_login_poll(api_url: str, client_id: str, device_code: str) -> Session | str:
    """Ask once whether the code has been approved.

    The cached Session on approval, otherwise the OAuth error code: 'authorization_pending' while
    the browser is still open, 'slow_down' to poll less often, and 'access_denied' or
    'expired_token' when no further poll can succeed.
    """
    answer = _token_request(api_url, {'grant_type': _DEVICE_GRANT, 'device_code': device_code, 'client_id': client_id})
    if isinstance(answer, TokenErrorResponse):
        return answer.code
    session = _create_session(answer, client_id)
    session_cache.save(api_url, session)
    return session


# Renewal spends the refresh token and writes back the rotated one; WorkOS refuses a token it has
# already rotated, so renewals run one at a time.
_renewal_lock = threading.Lock()


def access_token(api_url: str) -> str | None:
    """A token to send to api_url, renewing first if the cached one is spent. None when there is no session."""
    with _renewal_lock:
        session = session_cache.load(api_url)
        if session is None:
            return None
        if session.is_usable():
            return session.access_token
        if not session.can_refresh():
            raise excs.AuthorizationError(
                excs.ErrorCode.MISSING_CREDENTIALS, f'This Pixeltable session cannot be renewed. {SIGN_IN_AGAIN}'
            )

        # refresh token
        answer = _token_request(
            api_url,
            {
                'grant_type': 'refresh_token',
                'refresh_token': session.refresh_token or '',
                'client_id': session.client_id,
            },
        )
        if isinstance(answer, TokenErrorResponse):
            if answer.code == _REJECTED_GRANT:
                raise excs.AuthorizationError(
                    excs.ErrorCode.MISSING_CREDENTIALS,
                    f'Your Pixeltable session was rejected ({answer.code}: {answer.description}). {SIGN_IN_AGAIN}',
                )
            raise excs.AuthorizationError(
                excs.ErrorCode.MISSING_CREDENTIALS,
                f'Renewing your Pixeltable session failed ({answer.code}: {answer.description}). {SIGN_IN_AGAIN}',
            )
        renewed = _create_session(answer, session.client_id)
        if not renewed.email:
            renewed.email = session.email
        # WorkOS rotates on every renewal, so the token just sent is already spent: losing the new
        # one here would leave no way back into the session.
        session_cache.save(api_url, renewed)
        return renewed.access_token


def browser_logout_url(api_url: str) -> str:
    """WorkOS's sign-out URL for this session, or empty when the token has no session id.

    WorkOS directly, not the dashboard: a device-code sign-in never creates a dashboard session, so
    the dashboard's own sign-out has no cookie to clear.
    """
    session = session_cache.load(api_url)
    value = _claims(session.access_token).get('sid') if session is not None else ''
    session_id = value if isinstance(value, str) else ''
    if not session_id:
        return ''
    return f'{_WORKOS_LOGOUT}?session_id={urllib.parse.quote(session_id)}'
