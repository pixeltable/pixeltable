"""Signing in from the CLI, and keeping that session alive.

`pxt login` uses the OAuth device authorization grant (RFC 8628): the CLI asks WorkOS for a code,
you approve it in a browser, and the CLI polls until you have. Nothing secret travels through a URL,
and the CLI holds its own refresh token rather than a copy of the browser's session, so renewing on
one side cannot invalidate the other. Renewal is silent until MAX_SESSION_AGE_S.
"""

from __future__ import annotations

import base64
import binascii
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from typing import Any, Optional

from pixeltable.service import credentials
from pixeltable.service.credentials import Session

# Where a control plane says which WorkOS environment signs people in to it. Unauthenticated by
# necessity: a caller needs it before it can authenticate.
_AUTH_CONFIG_PATH = '/.well-known/pixeltable-auth'

# One WorkOS API for every environment; the client id is what tells them apart.
_WORKOS_API = 'https://api.workos.com'
_DEVICE_AUTH_PATH = '/user_management/authorize/device'
_TOKEN_PATH = '/user_management/authenticate'
_DEVICE_GRANT = 'urn:ietf:params:oauth:grant-type:device_code'

# Set when the cache alone proves the sign-in is over, so a caller can say so instead of guessing.
# WorkOS uses the second for a refresh token it has rejected, which means the same thing to a user.
SESSION_EXPIRED = 'session_expired'
_REJECTED_GRANT = 'invalid_grant'
NEEDS_SIGN_IN = (SESSION_EXPIRED, _REJECTED_GRANT)

_TIMEOUT_S = 30.0
# A ceiling on polling. WorkOS sets the real deadline in expires_in, which is usually shorter.
_LOGIN_TIMEOUT_S = 600.0


class AuthError(Exception):
    """Sign-in could not proceed. Carries the OAuth `error` code when the server supplied one.

    The code is what separates "keep polling" from "this failed", so it is kept apart from the text.
    """

    def __init__(self, message: str, code: str = '') -> None:
        super().__init__(message)
        self.code = code


def _request(url: str, data: Optional[bytes] = None, content_type: str = '') -> dict[str, Any]:
    """Fetch JSON. An error body becomes an AuthError; a bare HTTPError would discard it."""
    headers = {'Content-Type': content_type} if content_type else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            body = resp.read()
        try:
            payload = json.loads(body)
        except ValueError as e:
            # A 200 that is not JSON means something other than what we asked for answered -- an
            # environment too old to serve this, or a proxy in front of it.
            raise AuthError(f'{url} did not answer with JSON; is this environment up to date?') from e
        if not isinstance(payload, dict):
            raise AuthError(f'{url} answered with {type(payload).__name__}, not an object')
        return payload
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode())
        except Exception:
            raise AuthError(f'{url} returned HTTP {e.code}') from e
        code = str(payload.get('error') or '')
        detail = str(payload.get('error_description') or '') or f'HTTP {e.code}'
        raise AuthError(f'{code}: {detail}' if code else detail, code=code) from e
    except OSError as e:
        raise AuthError(f'could not reach {url}: {e}') from e


def _post_form(path: str, fields: dict[str, str]) -> dict[str, Any]:
    """POST to WorkOS. Form-encoded, and with no client secret: the CLI is a public client."""
    body = urllib.parse.urlencode(fields).encode()
    return _request(_WORKOS_API + path, data=body, content_type='application/x-www-form-urlencoded')


def auth_config(api_url: str) -> dict[str, Any]:
    """How to sign in to this control plane, asked of the control plane itself."""
    return _request(api_url.rstrip('/') + _AUTH_CONFIG_PATH)


def client_id_for(api_url: str) -> str:
    client_id = str(auth_config(api_url).get('client_id') or '')
    if not client_id:
        raise AuthError(f'{api_url} did not say which sign-in client to use')
    return client_id


def _expiry_from(token: str, default_s: float = 300.0) -> float:
    """The token's own `exp`, or a short default. Read, never verified: it only schedules renewal."""
    try:
        payload = token.split('.')[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + '=' * (-len(payload) % 4)))
        exp = claims.get('exp')
        if isinstance(exp, (int, float)):
            return float(exp)
    except (IndexError, ValueError, binascii.Error):
        pass
    return time.time() + default_s


def _number(payload: dict[str, Any], key: str, default: float) -> float:
    """A numeric field, or the default when it is absent or unusable. `or` would reject a real 0."""
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def _session_from(payload: dict[str, Any], client_id: str, logged_in_at: float) -> Session:
    token = str(payload.get('access_token') or '')
    if not token:
        raise AuthError('WorkOS returned no access token')
    user = payload.get('user') or {}
    return Session(
        access_token=token,
        expires_at=_expiry_from(token),
        refresh_token=str(payload.get('refresh_token') or ''),
        client_id=client_id,
        email=str(user.get('email') or ''),
        organization_id=str(payload.get('organization_id') or ''),
        logged_in_at=logged_in_at,
    )


def device_login(api_url: str, open_browser: bool = True) -> Session:
    """Sign in by approving a code in a browser, then cache the session. Returns it."""
    client_id = client_id_for(api_url)
    start = _post_form(_DEVICE_AUTH_PATH, {'client_id': client_id})

    user_code = str(start.get('user_code') or '')
    device_code = str(start.get('device_code') or '')
    verify = str(start.get('verification_uri_complete') or start.get('verification_uri') or '')
    if not (user_code and device_code and verify):
        raise AuthError('WorkOS did not return a device code')

    # stderr, not stdout: this is progress, and `pxt login --json` promises a parseable document.
    print(f'Your code is {user_code}', file=sys.stderr)
    print(f'Confirm it at {verify}', file=sys.stderr)
    if open_browser and not webbrowser.open(verify):
        print('Could not open a browser; open the link above.', file=sys.stderr)

    payload = _poll_for_approval(
        device_code, client_id, _number(start, 'expires_in', 300.0), _number(start, 'interval', 5.0)
    )
    session = _session_from(payload, client_id, logged_in_at=time.time())
    credentials.save(api_url, session)
    return session


def _poll_for_approval(device_code: str, client_id: str, expires_in: float, interval: float) -> dict[str, Any]:
    """Ask for the token until the code is approved, refused, or expires.

    WorkOS names both the interval and the deadline, so neither is guessed, and `slow_down` means
    back off rather than give up.
    """
    deadline = time.time() + min(expires_in, _LOGIN_TIMEOUT_S)
    while True:
        # Slept only as long as there is deadline left, and rechecked after: sleeping past expiry
        # and asking anyway spends a request the server has already stopped honouring.
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
        if time.time() >= deadline:
            break
        try:
            return _post_form(
                _TOKEN_PATH, {'grant_type': _DEVICE_GRANT, 'device_code': device_code, 'client_id': client_id}
            )
        except AuthError as e:
            if e.code == 'authorization_pending':
                continue
            if e.code == 'slow_down':
                # Five, per RFC 8628: less than that keeps polling faster than the server allows.
                interval += 5.0
                continue
            if e.code == 'access_denied':
                raise AuthError('the sign-in was refused in the browser') from e
            raise
    raise AuthError('the code expired before it was confirmed')


def refresh(api_url: str, session: Session, organization_id: str = '') -> Session:
    """Exchange the refresh token for a fresh one, persist the result, and return it.

    `organization_id` scopes the new token to that organization; without it WorkOS keeps whichever
    the session already had, which for an account that had none is still none.
    """
    if not session.can_refresh():
        raise AuthError('this session cannot be renewed', code=SESSION_EXPIRED)
    fields = {
        'grant_type': 'refresh_token',
        'refresh_token': session.refresh_token or '',
        'client_id': session.client_id,
    }
    if organization_id:
        fields['organization_id'] = organization_id
    payload = _post_form(_TOKEN_PATH, fields)
    renewed = _session_from(payload, session.client_id, logged_in_at=session.logged_in_at)
    if not renewed.email:
        renewed.email = session.email
    # WorkOS rotates the refresh token, so the one just sent is spent: saved before the caller uses
    # the new access token, not after.
    credentials.save(api_url, renewed)
    return renewed


# Control planes this process has already been authorized against. The deadline decides whether work
# may start, not whether work already running may finish: a bulk ingest that began inside the hour
# reconnects its tunnel after it, and failing there would lose the work rather than protect anything.
_authorized: set[str] = set()


def access_token(api_url: str) -> Optional[str]:
    """A token to send to `api_url`, renewing first if the cached one is spent.

    None when there is no session, so a caller can fall back to an API key; AuthError when a session
    exists but cannot be used.
    """
    session = credentials.load(api_url)
    if session is None:
        return None
    # Checked before the token's own expiry, and only once per process -- see _authorized.
    if api_url not in _authorized and session.is_expired():
        raise AuthError('your Pixeltable sign-in has expired', code=SESSION_EXPIRED)
    _authorized.add(api_url)
    if session.is_usable():
        return session.access_token
    return refresh(api_url, session).access_token


def authorize_org(api_url: str, org_id: str) -> Session:
    """Re-mint the cached session scoped to `org_id`. For an account that has just acquired one."""
    session = credentials.load(api_url)
    if session is None:
        raise AuthError('not signed in')
    return refresh(api_url, session, organization_id=org_id)
