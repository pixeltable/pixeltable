"""pcli test fixtures.

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
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@dataclass
class PcliResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def json(self) -> object:
        return json.loads(self.stdout)


@pytest.fixture(scope='session')
def pcli_daemon(init_env: None) -> Iterator[int]:
    port = _pick_port()
    env = {**os.environ, 'PCLI_PORT': str(port)}
    log_path = Path(tempfile.mkdtemp(prefix='pcli-test-')) / 'daemon.log'
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(
            [sys.executable, '-m', 'pcli.server.daemon'], env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL
        )
    try:
        from pcli.probe import is_running

        env_for_probe = os.environ.copy()
        os.environ['PCLI_PORT'] = str(port)
        deadline = time.time() + 15
        while time.time() < deadline:
            if is_running():
                break
            if proc.poll() is not None:
                tail = log_path.read_text(errors='replace')[-500:]
                raise RuntimeError(f'daemon exited early: {tail}')
            time.sleep(0.1)
        else:
            tail = log_path.read_text(errors='replace')[-500:]
            raise RuntimeError(f'daemon did not come up within 15s; log tail:\n{tail}')
        yield port
    finally:
        os.environ.clear()
        os.environ.update(env_for_probe)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def pcli(pcli_daemon: int, uses_db: None):
    def _run(*args: str, check: bool = True) -> PcliResult:
        env = {**os.environ, 'PCLI_PORT': str(pcli_daemon)}
        r = subprocess.run(['pcli', *args], capture_output=True, text=True, env=env)
        if check and r.returncode != 0:
            raise AssertionError(f'pcli {args} failed (rc={r.returncode}): {r.stderr}')
        return PcliResult(r.returncode, r.stdout, r.stderr)

    return _run
