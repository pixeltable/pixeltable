import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def test_sigterm_stops_the_daemon_at_once(tmp_path: Path) -> None:
    """In a container the daemon is PID 1, which ignores SIGTERM without a handler: a stopped catalog-server pod
    waited out its whole grace period and was killed."""
    port = _free_port()
    env = {**os.environ, 'PIXELTABLE_HOME': str(tmp_path / 'home'), 'PXT_PORT': str(port)}
    env.pop('PIXELTABLE_API_KEY', None)
    proc = subprocess.Popen(
        [sys.executable, '-m', 'pixeltable_cli.server.daemon', '--port', str(port), '--project-root', str(tmp_path)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 90
        while True:
            assert proc.poll() is None, 'the daemon exited before it served'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=2):
                    break
            except OSError:
                assert time.monotonic() < deadline, 'the daemon did not come up'
                time.sleep(0.2)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
