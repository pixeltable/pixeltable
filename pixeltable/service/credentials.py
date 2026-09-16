"""The session `pxt login` leaves behind, cached on this device.

Modelled on the AWS CLI's SSO cache: the browser sign-in happens once, the short-lived access token
it yields is written to disk, and every later command reuses it until it expires, renewing silently
rather than sending the user back to a browser. Nothing here is a long-lived credential -- an API key
is that, and it lives in the config file instead.

Two properties matter and are what the shape below is for:

  * a token is only valid against the control plane that issued it, so records are keyed by API URL.
    A prod session must never be presented to a sandbox, and a developer switching environments must
    not silently reuse the wrong identity.
  * the file holds bearer tokens, so it is created 0600 inside a 0700 directory and is never written
    in place -- a partial write would otherwise leave a truncated token that reads as a bad session.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from pixeltable.config import Config

# Refresh this long before the token actually expires, so a command that takes a moment to reach the
# control plane does not arrive with a token that died in flight.
EXPIRY_SKEW_S = 60.0

# How long a sign-in lasts before the browser is needed again, counted from `pxt login` and not reset
# by renewal. The token WorkOS issues outlives this by hours; a credential sitting on a laptop is a
# bearer token, and an unattended one should stop working the same day it was left behind. Anyone who
# wants a credential that does not expire wants an API key, which is what those are.
MAX_SESSION_AGE_S = 3600.0

_FILE_MODE = 0o600
_DIR_MODE = 0o700


@dataclasses.dataclass
class Session:
    """One signed-in identity, against one control plane."""

    access_token: str
    expires_at: float  # epoch seconds, read from the token's own exp claim
    # The renewable half. WorkOS does not hand out a raw refresh token: it returns a sealed session,
    # which the dashboard normally keeps as a cookie. A CLI has no cookie jar, so it holds the blob
    # and presents it to login_url to get a fresh access token.
    sealed_session: Optional[str] = None
    login_url: str = ''  # the dashboard that issued this session, and the only place it can be renewed
    email: str = ''  # for `pxt whoami`; never load-bearing, the token is what authorizes
    # When the browser sign-in happened. Carried through renewal unchanged, so renewing cannot walk
    # the deadline forward -- otherwise a session in daily use would never end.
    logged_in_at: float = 0.0

    def expires_in(self, now: Optional[float] = None) -> float:
        """Seconds left, negative once past expiry. Ignores the skew — that is a refresh decision."""
        return self.expires_at - (time.time() if now is None else now)

    def is_usable(self, now: Optional[float] = None) -> bool:
        """Whether this token can still be sent. False inside the skew window, so callers refresh."""
        return self.expires_in(now) > EXPIRY_SKEW_S

    def can_refresh(self) -> bool:
        return bool(self.sealed_session and self.login_url)

    def session_expires_in(self, now: Optional[float] = None) -> float:
        """Seconds until the browser is needed again. Negative once past it."""
        return self.logged_in_at + MAX_SESSION_AGE_S - (time.time() if now is None else now)

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Whether this sign-in is too old to keep renewing. Records from before this field existed
        carry logged_in_at == 0, which reads as expired: one re-login, rather than a session with no
        deadline at all."""
        return self.session_expires_in(now) <= 0


def _path() -> Path:
    return Config.get().home / 'credentials.json'


def _read_all() -> dict[str, Any]:
    path = _path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        # A corrupt cache is not an error worth failing a command over: it means "not signed in".
        return {}
    return data if isinstance(data, dict) else {}


def _write_all(sessions: dict[str, Any]) -> None:
    """Replace the cache atomically, 0600 inside a 0700 directory."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, _DIR_MODE)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.credentials-')
    try:
        os.fchmod(fd, _FILE_MODE)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def load(api_url: str) -> Optional[Session]:
    """The cached session for this control plane, expired or not. None when never signed in."""
    raw = _read_all().get('sessions', {}).get(api_url)
    if not isinstance(raw, dict):
        return None
    fields = {f.name for f in dataclasses.fields(Session)}
    try:
        return Session(**{k: v for k, v in raw.items() if k in fields})
    except TypeError:  # a record written by a version that required something we no longer have
        return None


def save(api_url: str, session: Session) -> None:
    data = _read_all()
    data.setdefault('sessions', {})[api_url] = dataclasses.asdict(session)
    _write_all(data)


def clear(api_url: Optional[str] = None) -> bool:
    """Forget one control plane's session, or every one. True when something was removed."""
    data = _read_all()
    sessions = data.get('sessions') or {}
    if api_url is None:
        removed = bool(sessions)
        data['sessions'] = {}
    else:
        removed = sessions.pop(api_url, None) is not None
    if removed:
        _write_all(data)
    return removed


def signed_in() -> list[str]:
    """The control plane URLs with a cached session, in order."""
    return sorted((_read_all().get('sessions') or {}).keys())
