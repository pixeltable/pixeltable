"""Stdlib-only so the client can import without pulling in pxt or pydantic."""

import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_PORT = 22089
_IS_WINDOWS = os.name == 'nt'


def _resolve_pixeltable_home() -> str:
    """Client-side mirror of pixeltable.config.Config's home-directory resolution"""
    return str(Path(os.environ.get('PIXELTABLE_HOME') or '~/.pixeltable').expanduser().resolve())


def _resolve_pixeltable_pgdata(home: str) -> str:
    """Mirrors env.py: PIXELTABLE_PGDATA if set, else {home}/pgdata."""
    return str(Path(os.environ.get('PIXELTABLE_PGDATA') or f'{home}/pgdata').expanduser().resolve())


def _resolve_pixeltable_config_file(home: str) -> str:
    """Mirrors config.py: PIXELTABLE_CONFIG if set, else {home}/config.toml."""
    return str(Path(os.environ.get('PIXELTABLE_CONFIG') or f'{home}/config.toml').expanduser().resolve())


def _daemon_log_path() -> str:
    return os.path.join(_resolve_pixeltable_home(), 'logs', 'pxt-daemon.log')


def get_port() -> int:
    return int(os.environ.get('PXT_PORT') or DEFAULT_PORT)


def pidfile_path() -> str:
    """Per-port pidfile path. The port parameterization isolates daemons running on
    different ports so they don't read or stomp each other's PID."""
    return os.path.join(_resolve_pixeltable_home(), f'pxt-daemon-{get_port()}.pid')


def read_pidfile() -> int | None:
    try:
        with open(pidfile_path(), encoding='utf-8') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def base_url() -> str:
    return f'http://127.0.0.1:{get_port()}'


def health_url() -> str:
    return f'{base_url()}/api/health'


# Identity fingerprint keys
_IDENTITY_KEYS: tuple[str, ...] = (
    'pxt_version',
    'pxt_install_dir',
    'python_executable',
    'pixeltable_home',
    'pixeltable_pgdata',
    'pixeltable_config_file',
    'pixeltable_env',
)


def fetch_health(timeout: float = 0.3) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(health_url(), timeout=timeout) as r:
            body = json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        return None
    # Verify this is actually our daemon and not some other service on the same port that
    # happens to return `{"ok": true}`. Require both the pxt service marker and the full
    # set of identity fields the mismatch / kill logic relies on.
    if not isinstance(body, dict) or body.get('service') != 'pxt' or not body.get('ok'):
        return None
    required = ('pid', 'started_at', *_IDENTITY_KEYS)
    if not all(k in body for k in required):
        return None
    return body


def is_running(timeout: float = 0.3) -> bool:
    return fetch_health(timeout) is not None


def _client_pxt_version() -> str | None:
    try:
        return importlib.metadata.version('pixeltable')
    except importlib.metadata.PackageNotFoundError:
        return None


def _client_pxt_install_dir() -> str | None:
    """Stdlib-only equivalent of `Path(pixeltable.__file__).parent`. Returns None if pixeltable isn't installed."""
    try:
        dist = importlib.metadata.distribution('pixeltable')
    except importlib.metadata.PackageNotFoundError:
        return None
    return str(Path(str(dist.locate_file('pixeltable'))).resolve())


# Env-var keys whose values may contain credentials and must not be returned verbatim
_SENSITIVE_NAME_PARTS: tuple[str, ...] = ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'PASSWD', 'CONNECT_STR')


def _is_sensitive_env_name(name: str) -> bool:
    upper = name.upper()
    return any(part in upper for part in _SENSITIVE_NAME_PARTS)


def _redact_env_value(name: str, value: str) -> str:
    if not _is_sensitive_env_name(name):
        return value
    digest = hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]
    return f'sha256:{digest}'


def _snapshot_pixeltable_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Returns dict mapping PIXELTABLE_* env vars to their redacted values."""
    env = os.environ if environ is None else environ
    return {k: _redact_env_value(k, env[k]) for k in sorted(env) if k.startswith('PIXELTABLE_')}


def identity() -> dict[str, Any]:
    pxt_version = _client_pxt_version()
    pxt_install_dir = _client_pxt_install_dir()
    # Surfacing this here turns a broken pixeltable install into one clear error instead
    # of a daemon that 500s on every /health call and respawns in a tight loop.
    if pxt_version is None or pxt_install_dir is None:
        raise RuntimeError(
            "pixeltable package metadata not found (importlib.metadata can't locate the "
            "'pixeltable' distribution). Reinstall with: pip install --force-reinstall pixeltable"
        )
    home = _resolve_pixeltable_home()
    return {
        'pxt_version': pxt_version,
        'pxt_install_dir': pxt_install_dir,
        'python_executable': sys.executable,
        'pixeltable_home': home,
        'pixeltable_pgdata': _resolve_pixeltable_pgdata(home),
        'pixeltable_config_file': _resolve_pixeltable_config_file(home),
        'pixeltable_env': _snapshot_pixeltable_env(),
    }


def _identity_diff(client: dict[str, Any], daemon: dict[str, Any]) -> list[str]:
    """Return the list of identity keys whose values differ."""
    return [k for k in _IDENTITY_KEYS if client.get(k) != daemon.get(k)]


def spawn_detached() -> None:
    log_path = _daemon_log_path()
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # POSIX: setsid() detaches from the controlling terminal; Windows: a new process
        # group + DETACHED_PROCESS gives the same "survive the parent shell" property.
        popen_kwargs: dict[str, Any] = {'stdin': subprocess.DEVNULL}
        if _IS_WINDOWS:
            popen_kwargs['creationflags'] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            )
        else:
            popen_kwargs['start_new_session'] = True
        with open(log_path, 'a', encoding='utf-8') as log:
            subprocess.Popen([sys.executable, '-m', 'pxt_cli.server.daemon'], stdout=log, stderr=log, **popen_kwargs)
    except OSError as e:
        reason = e.strerror or e.__class__.__name__
        raise RuntimeError(f'pxt daemon log unavailable ({log_path}): {reason}') from None


_TAIL_BYTES = 64 * 1024  # plenty of headroom for n_lines while bounding memory on huge logs


def _tail_daemon_log(n_lines: int = 10) -> str:
    try:
        with open(_daemon_log_path(), 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - _TAIL_BYTES))
            data = f.read()
    except OSError:
        return ''
    lines = data.decode('utf-8', errors='replace').splitlines()
    return '\n'.join(lines[-n_lines:]).rstrip()


def wait_for_health(timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_running():
            return
        time.sleep(0.1)
    tail = _tail_daemon_log()
    msg = f'pxt daemon did not come up within {timeout}s'
    if tail != '':
        msg += f'\n--- daemon log tail ---\n{tail}'
    raise RuntimeError(msg)


def _pid_alive(pid: int) -> bool:
    try:
        # signal 0 is the 'are you there?' probe (doesn't kill, just raises if the PID is gone)
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists but is owned by another user; treat as alive (we can't kill it anyway).
        return True
    except (OSError, SystemError):
        # SystemError surfaces on Windows when os.kill(pid, 0) hits an internal CPython
        # result-handling edge case for an unknown PID; treat the same as OSError ("gone").
        return False
    return True


def kill_and_wait(pid: int, timeout: float = 5.0) -> None:
    # Wait on the PID itself (not /health) so a hung-but-alive daemon that still holds the
    # listen socket is detected and SIGKILLed; otherwise the next spawn would fail with
    # 'address already in use' because we returned early on the health probe.
    # Windows has no SIGKILL; os.kill(pid, SIGTERM) calls TerminateProcess, which is
    # already a hard kill, so the fallback below is a no-op there.
    sigkill = getattr(signal, 'SIGKILL', signal.SIGTERM)
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, sigkill)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def ensure_running() -> str:
    health = fetch_health()
    if health is not None:
        client_identity = identity()
        diff = _identity_diff(client_identity, health)
        if len(diff) > 0:
            # Identity mismatch: the daemon was launched against a different install or env
            # snapshot than the client now sees. Restart, but only kill a PID we wrote
            # ourselves. We compare the pidfile against the responder's self-reported PID -
            # if they disagree, the responder is not our daemon and we refuse to SIGTERM an
            # unrelated process.
            tracked_pid = read_pidfile()
            reported_pid = health.get('pid')
            if tracked_pid is None or tracked_pid != reported_pid:
                raise RuntimeError(
                    f'a process on port {get_port()} is responding to /api/health but does not match '
                    f'our pidfile (pidfile={tracked_pid}, responder={reported_pid}); refusing to terminate it'
                )
            kill_and_wait(tracked_pid)
            spawn_detached()
            wait_for_health()
            # Cross-verify: the new responder must have a fresh PID and an identity that
            # fully matches the client. Anything else means the restart did not actually
            # swap in a daemon belonging to this install/env.
            new_health = fetch_health()
            if new_health is None:
                reason = 'new daemon did not respond to /pcli/v0/health'
            elif new_health.get('pid') == tracked_pid:
                reason = f'new daemon kept the killed PID {tracked_pid}'
            else:
                new_diff = _identity_diff(client_identity, new_health)
                if len(new_diff) > 0:
                    reason = f'new daemon still differs in: {", ".join(new_diff)}'
                else:
                    reason = ''
            if reason != '':
                raise RuntimeError(
                    f'pxt daemon restart did not produce a matching responder on port {get_port()}: {reason}'
                )
    else:
        spawn_detached()
        wait_for_health()
    return base_url()
