"""State the daemon keeps about itself: the requests it is serving and the config values it serves them with."""

from __future__ import annotations

import itertools
import logging
import os
import pathlib
import threading
import time
from typing import Mapping

import pydantic

from pixeltable.config import VAR_ENV_PREFIX, Config, env_var_name
from pixeltable_cli.client.utils import session_is_live
from pixeltable_cli.models import InFlightRequest, Method
from pixeltable_cli.utils import value_fingerprint

_logger = logging.getLogger(__name__)


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
    relevant = {name: h for name, h in other.items() if name.startswith(VAR_ENV_PREFIX) or name in known}
    missing = sorted(name for name in relevant if name not in mine)
    differing = sorted(name for name, h in relevant.items() if name in mine and mine[name] != h)
    return missing, differing


class _SavedWorkingDirs(pydantic.BaseModel):
    project_root: pathlib.Path | None
    working_dirs: dict[str, str]  # per session id


class DaemonState:
    """One daemon process's working directories, requests in flight and config values."""

    _in_flight: dict[int, InFlightRequest]
    _lock: threading.Lock
    _next_request_id: itertools.count
    _env_fingerprint: dict[str, str] | None  # None until the values have been recorded
    _wd: dict[str, str]  # working directory per session id
    _wd_file: str | None  # _save_wd() writes _wd here; None until load_wd()

    def __init__(self) -> None:
        self._in_flight = {}
        self._lock = threading.Lock()
        self._next_request_id = itertools.count()
        self._env_fingerprint = None
        self._wd = {}
        self._wd_file = None

    def load_wd(self, path: str) -> None:
        """Restore the working directories saved at path for this project, and save them there from now on."""
        wd: dict[str, str] = {}
        try:
            saved = _SavedWorkingDirs.model_validate_json(pathlib.Path(path).read_bytes())
            if saved.project_root == Config.get().project_root:
                wd = {session: uri for session, uri in saved.working_dirs.items() if session_is_live(session)}
        except FileNotFoundError:
            pass
        except (OSError, pydantic.ValidationError):
            _logger.warning('ignoring the working directories saved in %s', path, exc_info=True)
        with self._lock:
            self._wd = wd
            self._wd_file = path
            self._save_wd()

    def get_wd(self, session: str | None) -> str | None:
        """The working directory of the given session, if it set one."""
        if session is None:
            return None
        with self._lock:
            return self._wd.get(session)

    def set_wd(self, session: str, uri: str) -> None:
        with self._lock:
            self._wd[session] = uri
            self._save_wd()

    def clear_wd(self, session: str) -> None:
        with self._lock:
            self._wd.pop(session, None)
            self._save_wd()

    def _save_wd(self) -> None:
        """Write _wd to _wd_file. Call it with _lock acquired."""
        if self._wd_file is None:
            return
        tmp = f'{self._wd_file}.tmp'
        saved = _SavedWorkingDirs(project_root=Config.get().project_root, working_dirs=self._wd)
        try:
            pathlib.Path(tmp).write_text(saved.model_dump_json(), encoding='utf-8')
            os.replace(tmp, self._wd_file)
        except OSError:
            # the working directories still apply until the daemon restarts
            _logger.warning('cannot save the working directories to %s', self._wd_file, exc_info=True)

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
