"""Session-scoped credentials cache utilities"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import tempfile
import time
from typing import Any

from pixeltable.config import Config

# Treat a token as spent this long before it expires, so it cannot die in flight.
EXPIRY_SKEW_S = 60.0

_FILE_MODE = 0o600
_DIR_MODE = 0o700


@dataclasses.dataclass
class Session:
    """One signed-in identity, against one control plane."""

    access_token: str
    expires_at: float  # epoch seconds, from the token's own exp claim
    refresh_token: str | None = None  # rotated on every renewal
    client_id: str = ''  # the public client this session belongs to; renewal needs it
    email: str = ''  # shown by pxt whoami; nothing depends on it

    # The tenancy the token is scoped to. Empty for an account that has not finished onboarding,
    # which is worth saying plainly at sign-in rather than as a failure on the next command.
    organization_id: str = ''

    def expires_in(self, now: float | None = None) -> float:
        """Seconds left on the token, negative once past expiry."""
        return self.expires_at - (time.time() if now is None else now)

    def is_usable(self, now: float | None = None) -> bool:
        """Whether this token can still be sent. False inside the skew window."""
        return self.expires_in(now) > EXPIRY_SKEW_S

    def can_refresh(self) -> bool:
        return bool(self.refresh_token and self.client_id)


def _read_sessions() -> dict[str, Any]:
    """Every cached session, keyed by control plane. Empty before the first sign-in."""
    path = Config.get().home / 'auth' / 'sessions.json'
    if not path.is_file():
        return {}
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == _FILE_MODE
    data = json.loads(path.read_text(encoding='utf-8'))
    assert isinstance(data, dict)
    return data


def _write_sessions(cache: dict[str, Any]) -> None:
    path = Config.get().home / 'auth' / 'sessions.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, _DIR_MODE)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.sessions-')
    try:
        os.fchmod(fd, _FILE_MODE)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, sort_keys=True)
            # on disk before the rename, so a crash cannot leave a half-written file in its place
            f.flush()
            os.fsync(f.fileno())
        # replace() is atomic
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def load(api_url: str) -> Session | None:
    """The cached session for this control plane, expired or not. None when never signed in."""
    raw = _read_sessions().get(api_url)
    if raw is None:
        return None
    fields = {f.name for f in dataclasses.fields(Session)}
    try:
        return Session(**{k: v for k, v in raw.items() if k in fields})
    except TypeError:  # a record written by a version that required something we no longer have
        return None


def save(api_url: str, session: Session) -> None:
    cache = _read_sessions()
    cache[api_url] = dataclasses.asdict(session)
    _write_sessions(cache)


def clear(api_url: str | None = None) -> bool:
    """Forget one control plane's session, or every one. True when something was removed."""
    cache = _read_sessions()
    if api_url is None:
        removed = len(cache) > 0
        cache = {}
    else:
        removed = cache.pop(api_url, None) is not None
    if removed:
        _write_sessions(cache)
    return removed
