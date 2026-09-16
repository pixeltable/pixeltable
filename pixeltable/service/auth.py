"""Signing in from the CLI, and keeping that session alive.

`pxt login` opens the dashboard in a browser and gets the result back on a loopback listener. The
CLI cannot finish a WorkOS sign-in itself: that needs the WorkOS API key, which a distributed binary
must never hold, so the dashboard hands over the session instead. What it hands over is a sealed
session -- WorkOS exposes no raw refresh token -- which renews silently until MAX_SESSION_AGE_S.
"""

from __future__ import annotations

import base64
import binascii
import http.server
import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from typing import Any, ClassVar, Optional

from pixeltable.service import credentials
from pixeltable.service.credentials import Session

# Where a control plane says which dashboard signs people in to it. Unauthenticated by necessity.
_AUTH_CONFIG_PATH = '/.well-known/pixeltable-auth'
_CLI_LOGIN_PATH = '/api/auth/cli'
_CLI_TOKEN_PATH = '/api/auth/cli/token'

_TIMEOUT_S = 30.0
_LOGIN_TIMEOUT_S = 300.0
# 256 bits: the state proves a callback answers this login, not another local process's.
_STATE_BYTES = 32


class AuthError(Exception):
    """Sign-in could not proceed. Carries the OAuth `error` code when the server supplied one."""

    def __init__(self, message: str, code: str = '') -> None:
        super().__init__(message)
        self.code = code


def _get_json(url: str, headers: Optional[dict[str, str]] = None, data: Optional[bytes] = None) -> dict[str, Any]:
    """Fetch JSON. An error body becomes an AuthError; a bare HTTPError would discard it."""
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode())
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


def auth_config(api_url: str) -> dict[str, Any]:
    """How to sign in to this control plane, asked of the control plane itself."""
    return _get_json(api_url.rstrip('/') + _AUTH_CONFIG_PATH)


def login_url_for(api_url: str) -> str:
    url = str(auth_config(api_url).get('login_url') or '')
    if not url:
        raise AuthError(f'{api_url} did not say where to sign in')
    return url.rstrip('/')


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


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Receives the one redirect the dashboard sends back, and nothing else."""

    result: ClassVar[dict[str, str]] = {}

    def do_GET(self) -> None:  # BaseHTTPRequestHandler's interface names it this way
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        got = {k: v[0] for k, v in params.items() if v}
        type(self).result.update(got)
        body = (
            b'<html><body style="font-family:system-ui;padding:3rem">'
            b'<h2>Signed in.</h2><p>You can close this tab and return to your terminal.</p>'
            b'</body></html>'
        )
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        """Silence the default stderr access log; the CLI prints its own progress."""


def browser_login(api_url: str, open_browser: bool = True) -> Session:
    """Sign in through the dashboard and cache the session. Returns it."""
    login_url = login_url_for(api_url)
    state = secrets.token_urlsafe(_STATE_BYTES)

    # Loopback explicitly, never 0.0.0.0. Port 0 so concurrent sign-ins cannot collide.
    handler = type('_Handler', (_CallbackHandler,), {'result': {}})
    server = http.server.HTTPServer(('127.0.0.1', 0), handler)
    port = server.server_address[1]
    callback = f'http://127.0.0.1:{port}/callback'

    query = urllib.parse.urlencode({'callback': callback, 'state': state})
    target = f'{login_url}{_CLI_LOGIN_PATH}?{query}'

    # Listening before the browser opens: the reply is a redirect that arrives immediately.
    thread = threading.Thread(target=_serve_until_answered, args=(server, handler), daemon=True)
    thread.start()

    print(f'Opening {login_url} to sign in.')
    if open_browser:
        if not webbrowser.open(target):
            print(f'Could not open a browser. Open this URL instead:\n  {target}')
    else:
        print(f'Open this URL to sign in:\n  {target}')

    thread.join(timeout=_LOGIN_TIMEOUT_S)
    server.server_close()

    result = handler.result
    if not result:
        # A new account is sent to onboarding rather than back here, which looks like a closed tab.
        raise AuthError(
            'no reply from the browser. If you just created an account, finish setting up your '
            'organization in the dashboard and run `pxt login` again.'
        )
    # Before anything in the payload is read: a mismatch means this reply is not ours.
    if not secrets.compare_digest(result.get('state', ''), state):
        raise AuthError('the sign-in reply did not match this request; nothing was saved')
    sealed = result.get('session', '')
    if not sealed:
        raise AuthError(result.get('error') or 'the dashboard returned no session')

    token = result.get('access_token', '')
    session = Session(
        access_token=token,
        expires_at=_expiry_from(token) if token else 0.0,
        sealed_session=sealed,
        login_url=login_url,
        email=result.get('email', ''),
        logged_in_at=time.time(),
    )
    if not token:
        session = refresh(api_url, session)  # saves as a side effect
    else:
        credentials.save(api_url, session)
    return session


def _serve_until_answered(server: http.server.HTTPServer, handler: type) -> None:
    while not handler.result:
        server.handle_request()


def refresh(api_url: str, session: Session) -> Session:
    """Exchange the sealed session for a fresh access token, persist the result, and return it."""
    if not session.can_refresh():
        raise AuthError('this session cannot be renewed')
    payload = _get_json(
        session.login_url.rstrip('/') + _CLI_TOKEN_PATH,
        headers={'Authorization': f'Bearer {session.sealed_session}', 'Content-Type': 'application/json'},
        data=b'{}',  # a POST: the sealed session is a credential, not something to put in a URL
    )
    token = str(payload.get('token') or '')
    if not token:
        raise AuthError('the dashboard returned no token')
    renewed = Session(
        access_token=token,
        expires_at=_expiry_from(token),
        # The dashboard rotates this on renewal; the one we sent is now spent.
        sealed_session=str(payload.get('session') or session.sealed_session),
        login_url=session.login_url,
        email=session.email,
        logged_in_at=session.logged_in_at,
    )
    credentials.save(api_url, renewed)  # saved before use: the old session is spent
    return renewed


def access_token(api_url: str) -> Optional[str]:
    """A token to send to `api_url`, renewing first if the cached one is spent.

    None when there is no session, so a caller can fall back to an API key; AuthError when a session
    exists but cannot be used.
    """
    session = credentials.load(api_url)
    if session is None:
        return None
    # Before the token's own expiry: past the deadline, a still-valid token must not be sent.
    if session.is_expired():
        raise AuthError('your sign-in has expired')
    if session.is_usable():
        return session.access_token
    return refresh(api_url, session).access_token
