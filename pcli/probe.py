"""Stdlib-only so the client can import without pulling in pxt or pydantic."""

import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from pcli._paths import redact_home

DEFAULT_PORT = 22089
_IS_WINDOWS = os.name == 'nt'


def _daemon_log_path() -> str:
    """Resolve at call time so tests / users that set PIXELTABLE_HOME mid-run are honored."""
    home = os.environ.get('PIXELTABLE_HOME') or os.path.expanduser('~/.pixeltable')
    return os.path.join(home, 'logs', 'pcli-daemon.log')


def get_port() -> int:
    return int(os.environ.get('PCLI_PORT') or DEFAULT_PORT)


def pidfile_path() -> str:
    """Per-port pidfile: lets parallel pcli test workers (each on its own port) coexist."""
    home = os.environ.get('PIXELTABLE_HOME') or os.path.expanduser('~/.pixeltable')
    return os.path.join(home, f'pcli-daemon-{get_port()}.pid')


def _read_pidfile() -> int | None:
    try:
        with open(pidfile_path(), encoding='utf-8') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def base_url() -> str:
    return f'http://127.0.0.1:{get_port()}'


def health_url() -> str:
    return f'{base_url()}/pcli/v0/health'


def _fetch_health(timeout: float = 0.3) -> dict | None:
    try:
        with urllib.request.urlopen(health_url(), timeout=timeout) as r:
            body = json.loads(r.read())
        return body if body.get('ok') else None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        return None


def is_running(timeout: float = 0.3) -> bool:
    return _fetch_health(timeout) is not None


def _client_pxt_version() -> str | None:
    try:
        return importlib.metadata.version('pixeltable')
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_daemon_deps() -> None:
    """Fail fast if the optional `cli` deps aren't installed."""
    missing = [m for m in ('fastapi', 'uvicorn') if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(f'pcli daemon requires {", ".join(missing)} (install with: pip install pixeltable[cli])')


def spawn_detached() -> None:
    _check_daemon_deps()
    log_path = _daemon_log_path()
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # POSIX: setsid() detaches from the controlling terminal; Windows: a new process
        # group + DETACHED_PROCESS gives the same "survive the parent shell" property.
        popen_kwargs: dict = {'stdin': subprocess.DEVNULL}
        if _IS_WINDOWS:
            popen_kwargs['creationflags'] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            )
        else:
            popen_kwargs['start_new_session'] = True
        with open(log_path, 'a', encoding='utf-8') as log:
            subprocess.Popen([sys.executable, '-m', 'pcli.server.daemon'], stdout=log, stderr=log, **popen_kwargs)
    except OSError as e:
        # don't surface the resolved home path (which often appears in e.strerror); use the
        # redacted form so users see `$PIXELTABLE_HOME/logs/...` instead.
        reason = e.strerror or e.__class__.__name__
        raise RuntimeError(f'pcli daemon log unavailable ({redact_home(log_path)}): {reason}') from None


def _tail_daemon_log(n_lines: int = 10) -> str:
    try:
        with open(_daemon_log_path(), encoding='utf-8') as f:
            lines = f.readlines()
    except OSError:
        return ''
    return ''.join(lines[-n_lines:]).rstrip()


def wait_for_health(timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_running():
            return
        time.sleep(0.1)
    tail = _tail_daemon_log()
    msg = f'pcli daemon did not come up within {timeout}s'
    if tail:
        # Daemon log lines often embed resolved paths under PIXELTABLE_HOME; rewrite them
        # so user-facing CLI errors don't leak the operator's filesystem layout.
        msg += f'\n--- daemon log tail ---\n{redact_home(tail) or tail}'
    raise RuntimeError(msg)


def _kill_and_wait(pid: int, timeout: float = 5.0) -> None:
    # Windows has no SIGKILL; os.kill(pid, SIGTERM) calls TerminateProcess, which is
    # already a hard kill, so the fallback below is a no-op there.
    sigkill = getattr(signal, 'SIGKILL', signal.SIGTERM)
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_running():
            return
        time.sleep(0.1)
    try:
        os.kill(pid, sigkill)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def ensure_running() -> str:
    health = _fetch_health()
    if health is not None:
        client_ver = _client_pxt_version()
        if client_ver and health.get('pxt_version') and client_ver != health['pxt_version']:
            # version mismatch: restart, but only kill a PID we wrote ourselves. We compare
            # the pidfile against the responder's self-reported PID — if they disagree, the
            # responder is not our daemon and we refuse to SIGTERM an unrelated process.
            tracked_pid = _read_pidfile()
            reported_pid = health.get('pid')
            if tracked_pid is None or tracked_pid != reported_pid:
                raise RuntimeError(
                    f'a process on port {get_port()} is responding to /pcli/v0/health but does not match '
                    f'our pidfile (pidfile={tracked_pid}, responder={reported_pid}); refusing to terminate it'
                )
            _kill_and_wait(tracked_pid)
            spawn_detached()
            wait_for_health()
            # cross-verify: the responder should now be a different PID running our version
            new_health = _fetch_health()
            if (
                new_health is None
                or new_health.get('pid') == tracked_pid
                or (new_health.get('pxt_version') and new_health['pxt_version'] != client_ver)
            ):
                raise RuntimeError(
                    f'pcli daemon restart did not produce a matching-version responder on port {get_port()}'
                )
    else:
        spawn_detached()
        wait_for_health()
    return base_url()
