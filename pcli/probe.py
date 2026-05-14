"""Stdlib-only so the client can import without pulling in pxt or pydantic."""

import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PORT = 22089
_DAEMON_LOG_PATH = os.path.expanduser('~/.pixeltable/logs/pcli-daemon.log')


def get_port() -> int:
    return int(os.environ.get('PCLI_PORT') or DEFAULT_PORT)


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


def spawn_detached() -> None:
    os.makedirs(os.path.dirname(_DAEMON_LOG_PATH), exist_ok=True)
    with open(_DAEMON_LOG_PATH, 'a') as log:
        subprocess.Popen(
            [sys.executable, '-m', 'pcli.server.daemon'],
            start_new_session=True, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
        )


def _tail_daemon_log(n_lines: int = 10) -> str:
    try:
        with open(_DAEMON_LOG_PATH) as f:
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
        msg += f'\n--- daemon log tail ---\n{tail}'
    raise RuntimeError(msg)


def _kill_and_wait(pid: int, timeout: float = 5.0) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_running():
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def ensure_running() -> str:
    health = _fetch_health()
    if health is not None:
        client_ver = _client_pxt_version()
        if client_ver and health.get('pxt_version') and client_ver != health['pxt_version']:
            _kill_and_wait(health['pid'])
            spawn_detached()
            wait_for_health()
    else:
        spawn_detached()
        wait_for_health()
    return base_url()
