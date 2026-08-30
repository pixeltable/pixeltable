"""State the daemon keeps about itself: the requests it is serving and the config values it serves them with."""

from __future__ import annotations

import itertools
import threading
import time
from typing import Mapping

from pixeltable.config import SECRET_ENV_PREFIX, VAR_ENV_PREFIX, Config, ConfigKey, env_var_name
from pixeltable_cli.models import InFlightRequest, Method
from pixeltable_cli.utils import value_fingerprint


def config_fingerprint() -> dict[str, str]:
    """Fingerprints of the config values relevant to the daemon, as {env var name: hash}."""
    config = Config.get()
    out: dict[str, str] = {}
    for ck in config.env_keys():
        value: str | None
        # settings from [[pixeltable.database]] are read per command
        if (ck.section, ck.key) == ('pixeltable', 'database'):
            value = None
        else:
            value = config.get_value(ck.key, str, section=ck.section)
        if value is not None and value != '':
            out[env_var_name(ck.section, ck.key)] = value_fingerprint(value)
    return out


def compare_env_values(other: Mapping[str, str], mine: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """Compare the env fingerprint other with mine, as produced by config_fingerprint().

    Returns (set for other but not in mine, resolved differently in mine)
    """
    known = {env_var_name(ck.section, ck.key) for ck in Config.get().env_keys()}
    relevant = {
        name: h for name, h in other.items() if name.startswith((VAR_ENV_PREFIX, SECRET_ENV_PREFIX)) or name in known
    }
    missing = sorted(name for name in relevant if name not in mine)
    differing = sorted(name for name, h in relevant.items() if name in mine and mine[name] != h)
    return missing, differing


class DaemonState:
    """One daemon process's working directories, requests in flight and config values."""

    _in_flight: dict[int, InFlightRequest]
    _lock: threading.Lock
    _next_request_id: itertools.count
    _env_fingerprint: dict[str, str] | None  # None until the values have been recorded
    _wd: dict[str, str]  # working directory per session id

    def __init__(self) -> None:
        self._in_flight = {}
        self._lock = threading.Lock()
        self._next_request_id = itertools.count()
        self._env_fingerprint = None
        self._wd = {}

    def get_wd(self, session: str | None) -> str | None:
        """The working directory of the given session, if it set one."""
        if session is None:
            return None
        with self._lock:
            return self._wd.get(session)

    def set_wd(self, session: str, uri: str) -> None:
        with self._lock:
            self._wd[session] = uri

    def clear_wd(self, session: str) -> None:
        with self._lock:
            self._wd.pop(session, None)

    def begin_request(self, method: Method, path: str) -> int:
        """Register a request as in flight and return the id that end_request() takes."""
        request_id = next(self._next_request_id)
        with self._lock:
            self._in_flight[request_id] = InFlightRequest(method=method, path=path, started_at=time.time())
        return request_id

    def end_request(self, request_id: int) -> None:
        with self._lock:
            del self._in_flight[request_id]

    def in_flight_requests(self) -> list[InFlightRequest]:
        """The requests being served right now, oldest first."""
        with self._lock:
            return sorted(self._in_flight.values(), key=lambda r: r.started_at)

    def record_env_fingerprint(self) -> None:
        """Record the fingerprints of the config values this daemon serves with."""
        self._env_fingerprint = config_fingerprint()

    def known_env_vars(self) -> list[str]:
        """Every env var pixeltable reads config from, set or not."""
        return sorted(env_var_name(ck.section, ck.key) for ck in Config.get().env_keys())

    def changed_env_vars(self, current: Mapping[str, str]) -> list[str]:
        """The env vars whose value in current no longer matches the recorded _env_fingerprint."""
        if self._env_fingerprint is None:
            return []
        changed = {name for name, h in current.items() if self._env_fingerprint.get(name) != h}
        changed.update(name for name in self._env_fingerprint if name not in current)
        return sorted(changed)


state = DaemonState()
