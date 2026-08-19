"""State the daemon keeps about itself: the requests it is serving and the config values it serves them with."""

from __future__ import annotations

import itertools
import threading
import time

from pixeltable.config import Config, env_var_name
from pixeltable_cli.models import InFlightRequest, Method


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

    def ensure_env_fingerprint_recorded(self) -> None:
        """Record the fingerprints of the env-settable config values, unless they were recorded already."""
        if self._env_fingerprint is None:
            self._env_fingerprint = Config.get().env_fingerprint()

    def known_env_vars(self) -> list[str]:
        """Every env var pixeltable reads config from, set or not."""
        return sorted(env_var_name(ck.section, ck.key) for ck in Config.get().env_keys())

    def changed_env_vars(self) -> list[str]:
        """The env vars whose value no longer matches the recorded _env_fingerprint."""
        if self._env_fingerprint is None:
            return []
        current = Config.get().env_fingerprint()
        changed = {name for name, h in current.items() if self._env_fingerprint.get(name) != h}
        changed.update(name for name in self._env_fingerprint if name not in current)
        return sorted(changed)


state = DaemonState()
