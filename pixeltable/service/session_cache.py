"""Session-scoped credentials cache utilities"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fasteners import InterProcessLock  # type: ignore[import-untyped]

from pixeltable import exceptions as excs
from pixeltable.config import Config

# Treat a token as spent this long before it expires, so it cannot die in flight.
EXPIRY_SKEW_S = 60.0

_DIR_MODE = 0o700

# Windows has no POSIX permissions to set or check: it reports a writable file as 0o666, even after chmod().
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


# The types each Session field may have in a cache record.
_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    'access_token': (str,),
    'expires_at': (int, float),
    'refresh_token': (str, type(None)),
    'client_id': (str,),
    'email': (str,),
    'organization_id': (str,),
}


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
    """The sessions to carry over when the file is rewritten; none from an unreadable file, which is replaced."""
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
    """The session in one control plane's cache record; None when there is no record.

    Raises for a record that is not an object whose fields have the types in _FIELD_TYPES: a dataclass does
    not check the types of its fields, and a string in expires_at would fail the next renewal, not this read.
    """
    if record is None:
        return None
    if isinstance(record, dict):
        values = {f.name: record[f.name] for f in dataclasses.fields(Session) if f.name in record}
        # isinstance() accepts a bool as an int
        if all(isinstance(v, _FIELD_TYPES[k]) and not isinstance(v, bool) for k, v in values.items()):
            with contextlib.suppress(TypeError):  # the record lacks a field this version requires
                return Session(**values)
    raise _unusable('is unreadable')


def load(api_url: str) -> Session | None:
    """The cached session for this control plane, expired or not. None when never signed in.

    Raises when the file or this control plane's record in it is unreadable.
    """
    return _from_record(_read_sessions(check_private=True).get(api_url))


# InterProcessLock excludes other processes but not other threads of this one
_thread_lock = threading.Lock()


@contextlib.contextmanager
def _exclusive() -> Iterator[None]:
    """Hold the cache file for the whole of a read-modify-write, against this process's threads and other processes.

    Daemons for different projects share one Pixeltable home, so two writers would otherwise each read the
    old file and the later replace() would drop the earlier one's change. Neither lock is reentrant, so the
    holder must not call save(), clear() or renew().
    """
    with _thread_lock:
        path = Config.get().home / 'auth'
        path.mkdir(parents=True, exist_ok=True)
        with InterProcessLock(str(path / 'sessions.lock')):
            yield


class RejectedRefreshError(Exception):
    """Raised by the refresh() passed to renew() when the sign-in service refuses the refresh token.

    renew() discards the session, since a refused refresh token cannot renew it again, and then raises
    `error`.
    """

    error: excs.Error

    def __init__(self, error: excs.Error) -> None:
        super().__init__(error.message)
        self.error = error


def renew(api_url: str, stale: Callable[[Session], bool], refresh: Callable[[Session], Session]) -> Session | None:
    """The session for api_url, replaced by refresh(session) if stale(session) holds once the cache is locked.

    The lock is held across refresh(): WorkOS refuses a refresh token it has already rotated, so a second
    renewal of the same session would fail. Rereading under the lock picks up a renewal that finished
    meanwhile, and a sign-out, in which case this returns None.
    """
    with _exclusive():
        cache = _read_sessions(check_private=True)
        session = _from_record(cache.get(api_url))
        if session is None or not stale(session):
            return session
        try:
            renewed = refresh(session)
        except RejectedRefreshError as e:
            del cache[api_url]
            _write_sessions(cache)
            raise e.error from None
        # the refresh token just spent is gone, so losing the rotated one would leave no way back into the session
        cache[api_url] = dataclasses.asdict(renewed)
        _write_sessions(cache)
        return renewed


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
