"""pxt CLI test fixtures.

The daemon runs in a separate process, so it must inherit the per-worker
PIXELTABLE_* env vars set by the session-scoped init_env fixture. We spawn
our own daemon on a worker-specific port to avoid colliding with the user's
real daemon on 22089.
"""

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@dataclass
class PxtResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def json(self) -> Any:
        return json.loads(self.stdout)


@pytest.fixture(scope='session')
def pxt_daemon(init_env: None, tmp_path_factory: pytest.TempPathFactory) -> Iterator[int]:
    port = _pick_port()
    env = {**os.environ, 'PXT_PORT': str(port)}
    log_path = tmp_path_factory.mktemp('pxt-daemon') / 'daemon.log'
    prior_port = os.environ.get('PXT_PORT')
    with open(log_path, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(
            [sys.executable, '-m', 'pixeltable_cli.server.daemon'],
            env=env,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
        )
    try:
        from pixeltable_cli.client.utils import is_running

        os.environ['PXT_PORT'] = str(port)
        # Allow for a cold pixeltable import in the daemon subprocess, which on a loaded CI runner can run
        # well past a warm import; matches the client's own startup health timeout.
        startup_timeout = 45
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if is_running():
                break
            if proc.poll() is not None:
                tail = log_path.read_text(errors='replace')[-500:]
                raise RuntimeError(f'daemon exited early: {tail}')
            time.sleep(0.1)
        else:
            tail = log_path.read_text(errors='replace')[-500:]
            raise RuntimeError(f'daemon did not come up within {startup_timeout}s; log tail:\n{tail}')
        yield port
    finally:
        if prior_port is None:
            os.environ.pop('PXT_PORT', None)
        else:
            os.environ['PXT_PORT'] = prior_port
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


PxtRunner = Callable[..., PxtResult]


@pytest.fixture
def cli(pxt_daemon: int, uses_db: None) -> PxtRunner:
    def _run(*args: str, check: bool = True) -> PxtResult:
        # BROWSER=true prevents an actual browser tab open on `pxt dashboard` when tests are run on a dev machine.
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon), 'BROWSER': 'true'}
        r = subprocess.run(
            ['pxt', *args], capture_output=True, text=True, env=env, check=False, stdin=subprocess.DEVNULL
        )
        if check and r.returncode != 0:
            raise AssertionError(f'pxt {args} failed (rc={r.returncode}): {r.stderr}')
        return PxtResult(r.returncode, r.stdout, r.stderr)

    return _run
