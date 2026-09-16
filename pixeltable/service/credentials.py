"""The session `pxt login` leaves behind, cached on this device. Modelled on the AWS CLI's SSO cache.

Records are keyed by control-plane URL: a token is only valid against the one that issued it, and a
prod session must never reach a sandbox. The file holds bearer tokens, so it is 0600 inside a 0700
directory and replaced atomically. Nothing here is long-lived; an API key is that.
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

# Refresh this early, so a token cannot die in flight.
EXPIRY_SKEW_S = 60.0

# How long a sign-in lasts before the browser is needed again, counted from `pxt login` rather than
# from the last renewal. The token WorkOS issues outlives this by hours.
MAX_SESSION_AGE_S = 3600.0

_FILE_MODE = 0o600
_DIR_MODE = 0o700


@dataclasses.dataclass
class Session:
    """One signed-in identity, against one control plane."""

    access_token: str
    expires_at: float  # epoch seconds, from the token's own exp claim
    # The renewable half: WorkOS hands out no raw refresh token, only this sealed blob.
    sealed_session: Optional[str] = None
    login_url: str = ''  # the dashboard that issued this session, and the only place it renews
    email: str = ''  # for `pxt whoami`; never load-bearing
    logged_in_at: float = 0.0  # unchanged by renewal, so the deadline cannot walk forward

    def expires_in(self, now: Optional[float] = None) -> float:
        """Seconds left on the token, negative once past expiry."""
        return self.expires_at - (time.time() if now is None else now)

    def is_usable(self, now: Optional[float] = None) -> bool:
        """Whether this token can still be sent. False inside the skew window."""
        return self.expires_in(now) > EXPIRY_SKEW_S

    def can_refresh(self) -> bool:
        return bool(self.sealed_session and self.login_url)

    def session_expires_in(self, now: Optional[float] = None) -> float:
        """Seconds until the browser is needed again. Negative once past it."""
        return self.logged_in_at + MAX_SESSION_AGE_S - (time.time() if now is None else now)

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Whether this sign-in is too old to renew. A record predating logged_in_at reads as expired."""
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
        # A corrupt cache means "not signed in", not a failed command.
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
