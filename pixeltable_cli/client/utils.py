"""Client-only daemon orchestration: probe /api/health, spawn/kill/restart the daemon,
read the pidfile, tail the log on failed startup. Stdlib-only so importing this on every
`pxt` invocation stays cheap."""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from pixeltable_cli.utils import _IDENTITY_KEYS, _resolve_pixeltable_home, get_port, identity, pidfile_path

_IS_WINDOWS = os.name == 'nt'


def base_url() -> str:
    return f'http://127.0.0.1:{get_port()}'


def health_url() -> str:
    return f'{base_url()}/api/health'


def _daemon_log_path() -> str:
    """Per-port log path, matching the per-port pidfile."""
    return os.path.join(_resolve_pixeltable_home(), 'logs', f'pxt-daemon-{get_port()}.log')


def read_pidfile() -> int | None:
    try:
        with open(pidfile_path(), encoding='utf-8') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def fetch_health(timeout: float = 0.3) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(health_url(), timeout=timeout) as r:
            body = json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        return None
    # Verify this is actually our daemon and not some other service on the same port that
    # happens to return a JSON object with an ok=true field. Require both the pxt service
    # marker and the full set of identity fields the mismatch / kill logic relies on.
    if not isinstance(body, dict) or body.get('service') != 'pxt' or not body.get('ok'):
        return None
    required = ('pid', 'started_at', *_IDENTITY_KEYS)
    if not all(k in body for k in required):
        return None
    return body


def is_running(timeout: float = 0.3) -> bool:
    return fetch_health(timeout) is not None


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

        # `python -m` puts the daemon's working directory at the front of sys.path. If pxt is
        # invoked from a directory that contains a pixeltable/ or pixeltable_cli/ folder, that
        # folder shadows the installed package and the daemon imports the wrong code. Anchor the
        # daemon's cwd to the pixeltable home (which holds no importable packages) and set
        # PYTHONSAFEPATH so the working directory is not prepended to sys.path at all (3.11+).
        env = {**os.environ, 'PYTHONSAFEPATH': '1'}
        with open(log_path, 'a', encoding='utf-8') as log:
            subprocess.Popen(
                [sys.executable, '-m', 'pixeltable_cli.server.daemon'],
                stdout=log,
                stderr=log,
                cwd=_resolve_pixeltable_home(),
                env=env,
                **popen_kwargs,
            )
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


def _await_health(timeout: float) -> bool:
    """Poll /api/health until it responds or the timeout elapses. Returns whether it came up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_running():
            return True
        time.sleep(0.1)
    return False


def wait_for_health(timeout: float = 15.0) -> None:
    if _await_health(timeout):
        return
    tail = _tail_daemon_log()
    msg = f'pxt daemon did not come up within {timeout}s'
    if tail != '':
        msg += f'\n--- daemon log tail ---\n{tail}'
    raise RuntimeError(msg)


# A daemon that is starting up has already written its pidfile (the bind precedes the write) but
# does not serve /api/health until it finishes importing pixeltable, which is not cheap. Before
# treating a live-but-silent daemon as hung, give it this long to come up so we don't kill one
# that a concurrent `pxt` invocation just spawned.
_STARTUP_GRACE_PERIOD_SECS = 10.0


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
                reason = 'new daemon did not respond to /api/health'
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
        # Nothing is answering /api/health. Either no daemon is up, or one we started bound the
        # port and then wedged before serving health. The pidfile is our ownership record: if it
        # names a live process, that process is our daemon, so reclaim the port rather than let
        # spawn_detached() fail to bind it.
        stale_pid = read_pidfile()
        if stale_pid is not None and _pid_alive(stale_pid):
            # It may just be slow to start, so give it a grace window before concluding it is
            # hung; a daemon that comes up in the meantime is used as-is.
            if _await_health(_STARTUP_GRACE_PERIOD_SECS):
                return base_url()
            kill_and_wait(stale_pid)
        spawn_detached()
        wait_for_health()
    return base_url()
