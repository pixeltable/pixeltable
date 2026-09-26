"""The on-disk cache of credentials, one per control plane: a `pxt login` session or a `pxt new` trial."""

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


@dataclasses.dataclass
class Trial:
    """A trial organization from `pxt new`, against one control plane. Its API key is never renewed."""

    api_key: str
    org: str
    org_id: str
    db: str
    claim_url: str  # whoever opens it becomes the organization's admin
    expires_at: float  # epoch seconds; unclaimed, the organization is deleted then

    def is_expired(self, now: float | None = None) -> bool:
        return self.expires_at <= (time.time() if now is None else now)


# The types each Session field may have in a cache record.
_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    'access_token': (str,),
    'expires_at': (int, float),
    'refresh_token': (str, type(None)),
    'client_id': (str,),
    'email': (str,),
    'organization_id': (str,),
}

# The types each Trial field may have in a cache record.
_TRIAL_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    'api_key': (str,),
    'org': (str,),
    'org_id': (str,),
    'db': (str,),
    'claim_url': (str,),
    'expires_at': (int, float),
}

# A trial's record has this value under 'kind'. A session's record has no 'kind': versions before `pxt new` wrote
# none, and they reject a record that lacks access_token, so none of them sends a trial's key as a session token.
_TRIAL_KIND = 'trial'


def _path() -> Path:
    return Config.get().home / 'auth' / 'sessions.json'


def _unusable(reason: str) -> excs.Error:
    return excs.AuthorizationError(
        excs.ErrorCode.MISSING_CREDENTIALS,
        f'Your cached Pixeltable sign-in {reason}, so it was not used. Run `pxt logout`, then `pxt login`.',
    )


def _read_sessions(*, check_private: bool) -> dict[str, Any]:
    """Every cached session, keyed by control plane. Empty before the first sign-in.

    Raises for a file that this user cannot read or that is not a JSON object, and with check_private for
    one that another user owns or can read.
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
    except PermissionError:
        raise _unusable('is unreadable') from None
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise _unusable('is unreadable')
    return data


def _rewritable_sessions() -> dict[str, Any]:
    """The sessions to carry over when the file is rewritten.

    Empty for a file that is unreadable, or that another user owns or can read: another user may have
    planted or copied its tokens, which must not become usable by being rewritten into a private file.
    """
    try:
        return _read_sessions(check_private=True)
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


def _to_record(credential: Session | Trial) -> dict[str, Any]:
    record = dataclasses.asdict(credential)
    if isinstance(credential, Trial):
        record['kind'] = _TRIAL_KIND
    return record


def _from_record(record: Any) -> Session | Trial | None:
    """The credential in one control plane's cache record; None when there is no record.

    Raises for a record that is not an object whose fields have the types in _FIELD_TYPES, or in
    _TRIAL_FIELD_TYPES for a trial: a dataclass does not check the types of its fields, and a string in
    expires_at would fail the next renewal, not this read.
    """
    if record is None:
        return None
    if isinstance(record, dict) and ('kind' not in record or record['kind'] == _TRIAL_KIND):
        cls: type[Session | Trial] = Trial if 'kind' in record else Session
        field_types = _TRIAL_FIELD_TYPES if cls is Trial else _FIELD_TYPES
        values = {f.name: record[f.name] for f in dataclasses.fields(cls) if f.name in record}
        # isinstance() accepts a bool as an int
        if all(isinstance(v, field_types[k]) and not isinstance(v, bool) for k, v in values.items()):
            with contextlib.suppress(TypeError):  # the record lacks a field this version requires
                return cls(**values)
    raise _unusable('is unreadable')


def load_credential(api_url: str) -> Session | Trial | None:
    """The cached session or trial for this control plane, expired or not. None when there is neither.

    Raises when the file or this control plane's record in it is unreadable.
    """
    return _from_record(_read_sessions(check_private=True).get(api_url))


def load(api_url: str) -> Session | None:
    """The cached session for this control plane, expired or not. None when never signed in, or for a trial.

    Raises when the file or this control plane's record in it is unreadable.
    """
    cached = load_credential(api_url)
    return cached if isinstance(cached, Session) else None


def load_for_sign_out(api_url: str) -> Session | None:
    """The cached session for this control plane, even from a file that other users can read.

    None when never signed in, for a trial, and when the file or this control plane's record in it is
    unreadable. The session's token must not be sent: another user may have copied it.
    """
    try:
        cached = _from_record(_read_sessions(check_private=False).get(api_url))
    except excs.AuthorizationError:
        return None
    return cached if isinstance(cached, Session) else None


# InterProcessLock excludes other processes but not other threads of this one
_thread_lock = threading.Lock()


@contextlib.contextmanager
def _exclusive() -> Iterator[None]:
    """Hold the cache file for the whole of a read-modify-write, against this process's threads and other processes.

    Daemons for different projects share one Pixeltable home, so two writers would otherwise each read the
    old file and the later replace() would drop the earlier one's change. Neither lock is reentrant, so the
    holder must not call save(), clear(), renew() or reuse_or_create_trial().
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
    meanwhile, and a sign-out or a trial in the session's place, in which case this returns None.
    """
    with _exclusive():
        cache = _read_sessions(check_private=True)
        session = _from_record(cache.get(api_url))
        if not isinstance(session, Session):
            return None
        if not stale(session):
            return session
        try:
            renewed = refresh(session)
        except RejectedRefreshError as e:
            del cache[api_url]
            _write_sessions(cache)
            raise e.error from None
        # the refresh token just spent is gone, so losing the rotated one would leave no way back into the session
        cache[api_url] = _to_record(renewed)
        _write_sessions(cache)
        return renewed


def save(api_url: str, credential: Session | Trial) -> None:
    with _exclusive():
        cache = _rewritable_sessions()
        cache[api_url] = _to_record(credential)
        _write_sessions(cache)


def reuse_or_create_trial(api_url: str, create: Callable[[], Trial]) -> tuple[Session | Trial, bool]:
    """The session or unexpired trial cached for api_url, else the trial create() returns, cached in its place.

    True with the trial create() returned. The lock is held across create(), so that concurrent callers create one
    trial: the site hands out a trial's API key once, and a second trial would replace the first one's record.
    Raises when the file or this control plane's record in it is unreadable, rather than replace it.
    """
    with _exclusive():
        cache = _read_sessions(check_private=True)
        cached = _from_record(cache.get(api_url))
        if isinstance(cached, Session) or (isinstance(cached, Trial) and not cached.is_expired()):
            return cached, False
        trial = create()
        cache[api_url] = _to_record(trial)
        _write_sessions(cache)
        return trial, True


def clear(api_url: str | None = None) -> bool:
    """Forget one control plane's session, or every one. True when something was removed.

    A file that is unreadable is deleted outright, as no single session can be removed from it. So is one
    that another user owns or can read, whose sessions must not be rewritten (see _rewritable_sessions()).
    """
    with _exclusive():
        try:
            cache = _read_sessions(check_private=True)
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
