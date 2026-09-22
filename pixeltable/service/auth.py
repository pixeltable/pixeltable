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

# Where a control plane says which WorkOS environment signs people in to it; served without
# authentication.
_AUTH_CONFIG_PATH = '/.well-known/pixeltable-auth'

# What an error calls it: the user never chose WorkOS and cannot act on its name.
_SIGN_IN_SERVICE = 'the Pixeltable sign-in service'
_DEVICE_GRANT = 'urn:ietf:params:oauth:grant-type:device_code'

# AuthKit CLI auth lives on the WorkOS API, not on the AuthKit domain: the /oauth2 endpoints that
# domain advertises through OIDC discovery serve Connect apps, and a public CLI client is not one.
# https://workos.com/docs/authkit/cli-auth
_WORKOS_API = 'https://api.workos.com'
_DEVICE_AUTH_PATH = '/user_management/authorize/device'
_TOKEN_PATH = '/user_management/authenticate'
_LOGOUT_PATH = '/user_management/sessions/logout'

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
        resp = SESSION.post(sign_in_config(api_url).url(_TOKEN_PATH), data=fields, timeout=_TIMEOUT_S)
    except requests.RequestException as e:
        raise _unreachable(_SIGN_IN_SERVICE, e) from e
    if resp.status_code in (200, 201):
        return _payload(_SIGN_IN_SERVICE, resp)
    # a 5xx means the service broke, not that it decided something about this request
    if resp.status_code >= 500:
        raise _bad_status(_SIGN_IN_SERVICE, resp)
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
    """One control plane's sign-in service, and the public client to present to it."""

    client_id: str
    workos_api: str

    def url(self, path: str) -> str:
        return self.workos_api + path


_config_cache: dict[str, SignInConfig] = {}  # key: API URL
_config_lock = threading.Lock()


def sign_in_config(api_url: str) -> SignInConfig:
    """Read this control plane's public client and its WorkOS API into _config_cache.

    workos_api is optional: every environment reaches the same WorkOS today, so a control plane that
    states none gets the default. A test control plane sets it to point the flow at itself.
    """
    with _config_lock:
        cached = _config_cache.get(api_url)
        if cached is not None:
            return cached
        config = _json_request(api_url.rstrip('/') + _AUTH_CONFIG_PATH, api_url)
        client_id = str(config.get('client_id') or '')
        if client_id == '':
            raise excs.InternalError(
                excs.ErrorCode.INTERNAL_ERROR, f'{api_url} did not say which sign-in client to use'
            )
        resolved = SignInConfig(client_id, str(config.get('workos_api') or _WORKOS_API).rstrip('/'))
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
    claims = _claims(token)
    exp = claims.get('exp')
    expires_at = float(exp) if isinstance(exp, (int, float)) else time.time() + 300.0
    # the control plane reads the organization from the token's org_id claim
    org_claim = claims.get('org_id')
    return Session(
        access_token=token,
        expires_at=expires_at,
        refresh_token=str(payload.get('refresh_token') or ''),
        client_id=client_id,
        email=str(user.get('email') or ''),
        organization_id=str(payload.get('organization_id') or (org_claim if isinstance(org_claim, str) else '')),
    )


def device_login_start(api_url: str) -> LoginStartResponse:
    """Start the device flow. This returns as soon as the code exists: no browser, no waiting."""
    resolved = sign_in_config(api_url)
    client_id = resolved.client_id
    authz_resp = _json_request(resolved.url(_DEVICE_AUTH_PATH), _SIGN_IN_SERVICE, {'client_id': client_id})
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


def _refresh(api_url: str, session: Session, organization_id: str) -> Session:
    """Spend the session's refresh token on a new token, scoped to organization_id unless it is empty.

    WorkOS keeps the token's organization when the request omits organization_id.
    """
    if not session.can_refresh():
        raise excs.AuthorizationError(
            excs.ErrorCode.MISSING_CREDENTIALS, f'This Pixeltable session cannot be renewed. {SIGN_IN_AGAIN}'
        )
    fields = {
        'grant_type': 'refresh_token',
        'refresh_token': session.refresh_token or '',
        'client_id': session.client_id,
    }
    if organization_id != '':
        fields['organization_id'] = organization_id
    answer = _token_request(api_url, fields)
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
    renewed.organization_id = renewed.organization_id or organization_id
    renewed.email = renewed.email or session.email
    return renewed


def access_token(api_url: str) -> str | None:
    """A token to send to api_url, renewing first if the cached one is spent. None when there is no session."""
    with _renewal_lock:
        session = session_cache.load(api_url)
        if session is None:
            return None
        if session.is_usable():
            return session.access_token
        renewed = _refresh(api_url, session, session.organization_id)
        # WorkOS rotates on every renewal, so the token just sent is already spent: losing the new
        # one here would leave no way back into the session.
        session_cache.save(api_url, renewed)
        return renewed.access_token


def rescope(api_url: str, organization_id: str) -> Session:
    """Renew the session for organization_id, and return the renewed session.

    This renews a usable token too: one issued before the organization existed is scoped to no organization.
    """
    with _renewal_lock:
        session = session_cache.load(api_url)
        if session is None:
            raise excs.AuthorizationError(
                excs.ErrorCode.MISSING_CREDENTIALS, f'There is no Pixeltable session to switch. {SIGN_IN_AGAIN}'
            )
        renewed = _refresh(api_url, session, organization_id)
        session_cache.save(api_url, renewed)
        return renewed


def browser_logout_url(api_url: str, session: Session) -> str:
    """WorkOS's sign-out URL for this session, or empty when the token has no session id.

    WorkOS directly, not the dashboard: a device-code sign-in never creates a dashboard session, so
    the dashboard's own sign-out has no cookie to clear.
    """
    value = _claims(session.access_token).get('sid')
    session_id = value if isinstance(value, str) else ''
    if not session_id:
        return ''
    logout = sign_in_config(api_url).url(_LOGOUT_PATH)
    return f'{logout}?session_id={urllib.parse.quote(session_id)}'
