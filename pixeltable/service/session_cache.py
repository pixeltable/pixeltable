"""Session-scoped credentials cache utilities"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

from fasteners import InterProcessLock  # type: ignore[import-untyped]

from pixeltable import exceptions as excs
from pixeltable.config import Config

# Treat a token as spent this long before it expires, so it cannot die in flight.
EXPIRY_SKEW_S = 60.0

_DIR_MODE = 0o700

# Windows has no POSIX permissions to set or check: it reports every file as 0o666, even after chmod().
_POSIX = os.name == 'posix'


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


def _path() -> Path:
    return Config.get().home / 'auth' / 'sessions.json'


def _unusable(reason: str) -> excs.Error:
    return excs.AuthorizationError(
        excs.ErrorCode.MISSING_CREDENTIALS,
        f'Your cached Pixeltable sign-in {reason}, so it was not used. Run `pxt logout`, then `pxt login`.',
    )


def _read_sessions(*, check_private: bool) -> dict[str, Any]:
    """Every cached session, keyed by control plane. Empty before the first sign-in.

    Raises for a file that is not a JSON object, and with check_private for one that another user owns
    or can read.
    """
    try:
        with open(_path(), 'rb') as f:
            if check_private and _POSIX:
                st = os.fstat(f.fileno())
                if stat.S_IMODE(st.st_mode) & 0o077 != 0 or st.st_uid != os.getuid():
                    raise _unusable('can be read by other users')
            raw = f.read()
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise _unusable('is unreadable')
    return data


def _rewritable_sessions() -> dict[str, Any]:
    """The sessions to carry into a rewrite of the file, which replaces an unreadable one."""
    try:
        return _read_sessions(check_private=False)
    except excs.AuthorizationError:
        return {}


def _write_sessions(cache: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if _POSIX:
        os.chmod(path.parent, _DIR_MODE)
    # mkstemp() creates the file readable and writable by its owner only
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.sessions-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, sort_keys=True)
            # on disk before the rename, so a crash cannot leave a half-written file in its place
            f.flush()
            os.fsync(f.fileno())
        # replace() is atomic
        os.replace(tmp, path)
    except BaseException:
        # the with block has closed the file by now; Windows cannot delete an open one
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _from_record(record: Any) -> Session | None:
    if not isinstance(record, dict):
        return None
    fields = {f.name for f in dataclasses.fields(Session)}
    try:
        return Session(**{k: v for k, v in record.items() if k in fields})
    except TypeError:  # the record lacks a field this version requires
        return None


def load(api_url: str) -> Session | None:
    """The cached session for this control plane, expired or not. None when never signed in."""
    return _from_record(_read_sessions(check_private=True).get(api_url))


def _exclusive() -> InterProcessLock:
    """The lock a writer takes for the whole of its read-modify-write.

    Daemons for different projects share one Pixeltable home, so two renewals for different control
    planes would otherwise each read the old file and the later replace() would drop the earlier one.
    """
    path = Config.get().home / 'auth'
    path.mkdir(parents=True, exist_ok=True)
    return InterProcessLock(str(path / 'sessions.lock'))


def save(api_url: str, session: Session) -> None:
    with _exclusive():
        cache = _rewritable_sessions()
        cache[api_url] = dataclasses.asdict(session)
        _write_sessions(cache)


def clear(api_url: str | None = None) -> bool:
    """Forget one control plane's session, or every one. True when something was removed.

    An unreadable file is deleted outright, since no single session in it can be removed.
    """
    with _exclusive():
        try:
            cache = _read_sessions(check_private=False)
        except excs.AuthorizationError:
            _path().unlink(missing_ok=True)
            return True
        if api_url is None:
            removed = len(cache) > 0
            cache = {}
        else:
            removed = cache.pop(api_url, None) is not None
        if removed:
            _write_sessions(cache)
    return removed
