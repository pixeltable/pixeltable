"""Stdlib-only so the client can import without pulling in pxt or pydantic."""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PORT = 22089


def get_port() -> int:
    return int(os.environ.get('PCLI_PORT') or DEFAULT_PORT)


def health_url() -> str:
    return f'http://localhost:{get_port()}/pcli/v0/health'


def is_running(timeout: float = 0.3) -> bool:
    try:
        with urllib.request.urlopen(health_url(), timeout=timeout) as r:
            return bool(json.loads(r.read()).get('ok'))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        return False


def spawn_detached() -> None:
    log_path = os.path.expanduser('~/.pixeltable/logs/pcli-daemon.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, 'a')
    subprocess.Popen(
        [sys.executable, '-m', 'pcli.server.daemon'],
        start_new_session=True, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
    )


def wait_for_health(timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_running():
            return
        time.sleep(0.1)
    raise RuntimeError(f'pcli daemon did not come up within {timeout}s')


def ensure_running() -> str:
    if not is_running():
        spawn_detached()
        wait_for_health()
    return f'http://localhost:{get_port()}'
