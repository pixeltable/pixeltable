"""Tests for probe.ensure_running() spawn and version-mismatch restart paths.

The smoke tests always pre-spawn the daemon via the `pcli_daemon` fixture, so the
auto-spawn path and the pidfile-guarded restart logic aren't exercised there. These
tests fill that gap:

  - one integration test that runs `pcli` against a port with no daemon running and
    asserts the client auto-spawned a working one;
  - three unit-style tests that monkeypatch the probe internals to exercise the
    restart safety logic without touching real processes.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest

# The daemon requires the `cli` extra; skip the whole module on `minimal` installs.
pytest.importorskip('fastapi')
pytest.importorskip('uvicorn')

from pcli import probe


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def fresh_port(init_env: None) -> Iterator[int]:
    """Allocate a port that no daemon is currently using and clean up afterward."""
    port = _pick_port()
    prior = os.environ.get('PCLI_PORT')
    os.environ['PCLI_PORT'] = str(port)
    try:
        yield port
    finally:
        pid = probe._read_pidfile()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
                # give the daemon a moment to exit so its pidfile gets removed
                for _ in range(30):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.1)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        if prior is None:
            os.environ.pop('PCLI_PORT', None)
        else:
            os.environ['PCLI_PORT'] = prior


class TestProbe:
    def test_auto_spawn_when_no_daemon_running(self, fresh_port: int) -> None:
        """Cold start: no daemon on the port, the pcli client spawns one and routes the command."""
        env = {**os.environ, 'PCLI_PORT': str(fresh_port)}
        r = subprocess.run(
            [sys.executable, '-m', 'pcli.client.main', 'health'],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=30,
        )
        assert r.returncode == 0, f'pcli health failed (rc={r.returncode}): {r.stderr}'
        body = json.loads(r.stdout)
        assert body['service'] == 'pcli'
        assert body['ok'] is True
        assert body['pid'] > 0
        # the spawned daemon's pidfile should now exist and contain that PID
        assert probe._read_pidfile() == body['pid']

    def test_version_mismatch_refuses_unknown_pid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Safety: if a responder reports a different version AND its PID doesn't match our
        pidfile, refuse to SIGTERM it. It might be an unrelated process on the same port."""
        foreign_responder = {
            'ok': True,
            'service': 'pcli',
            'pxt_version': 'OLD',
            'pid': 99999,
            'started_at': '2026-01-01',
        }
        monkeypatch.setattr(probe, '_fetch_health', lambda *a, **kw: foreign_responder)
        monkeypatch.setattr(probe, '_client_pxt_version', lambda: 'NEW')
        monkeypatch.setattr(probe, '_read_pidfile', lambda: 12345)
        killed: list = []
        monkeypatch.setattr(probe, '_kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        monkeypatch.setattr(probe, 'spawn_detached', lambda: killed.append('spawn'))

        with pytest.raises(RuntimeError, match='does not match our pidfile'):
            probe.ensure_running()
        assert killed == []

    def test_version_mismatch_restart_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy restart: matching pidfile + version mismatch causes kill, spawn, and a
        post-restart cross-verify against the new responder."""
        responses = iter(
            [
                {'ok': True, 'service': 'pcli', 'pxt_version': 'OLD', 'pid': 100, 'started_at': 'a'},
                {'ok': True, 'service': 'pcli', 'pxt_version': 'NEW', 'pid': 200, 'started_at': 'b'},
            ]
        )
        monkeypatch.setattr(probe, '_fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(probe, '_client_pxt_version', lambda: 'NEW')
        monkeypatch.setattr(probe, '_read_pidfile', lambda: 100)
        actions: list = []
        monkeypatch.setattr(probe, '_kill_and_wait', lambda pid, timeout=5.0: actions.append(('kill', pid)))
        monkeypatch.setattr(probe, 'spawn_detached', lambda: actions.append(('spawn',)))
        monkeypatch.setattr(probe, 'wait_for_health', lambda timeout=15.0: None)

        url = probe.ensure_running()
        assert url.startswith('http://127.0.0.1:')
        assert actions == [('kill', 100), ('spawn',)]

    def test_version_mismatch_restart_verify_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Post-restart cross-verify: if the new responder still reports the killed PID,
        fail loudly instead of silently routing to whatever it is."""
        responses = iter(
            [
                {'ok': True, 'service': 'pcli', 'pxt_version': 'OLD', 'pid': 100, 'started_at': 'a'},
                {'ok': True, 'service': 'pcli', 'pxt_version': 'NEW', 'pid': 100, 'started_at': 'a'},
            ]
        )
        monkeypatch.setattr(probe, '_fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(probe, '_client_pxt_version', lambda: 'NEW')
        monkeypatch.setattr(probe, '_read_pidfile', lambda: 100)
        monkeypatch.setattr(probe, '_kill_and_wait', lambda pid, timeout=5.0: None)
        monkeypatch.setattr(probe, 'spawn_detached', lambda: None)
        monkeypatch.setattr(probe, 'wait_for_health', lambda timeout=15.0: None)

        with pytest.raises(RuntimeError, match='did not produce a matching-version responder'):
            probe.ensure_running()
