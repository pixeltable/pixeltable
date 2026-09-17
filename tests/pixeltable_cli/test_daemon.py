"""Tests for how a cli client spawns or adopts its daemon."""

import contextlib
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import psutil
import pytest

from pixeltable_cli.utils import pidfile_path

_HEALTH_TIMEOUT_SECS = 180.0


@pytest.fixture
def daemon_port(init_env: None) -> Iterator[int]:
    """Picks an available port to use for a daemon. Runs pxt daemon stop after the test."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    yield port
    subprocess.run(
        ['pxt', 'daemon', 'stop', '-f'],
        env={**os.environ, 'PXT_PORT': str(port)},
        capture_output=True,
        check=False,
        timeout=60,
    )


def _create_pxt_project(root: pathlib.Path) -> None:
    """Creates an empty project at root by placing an empty marker file in it."""
    root.mkdir()
    (root / 'pixeltable.toml').write_text('', encoding='utf-8')


def _pxt_health(port: int, cwd: pathlib.Path, env_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """Runs `pxt health` and returns its parsed json output. If the daemon is not yet running, `pxt health` will start
    one, and it inherits this environment."""
    r = subprocess.run(
        ['pxt', 'health'],
        env={**os.environ, 'PXT_PORT': str(port), **(env_overrides or {})},
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=_HEALTH_TIMEOUT_SECS,
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


class TestDaemon:
    def test_replace_another_projects_daemon(self, daemon_port: int, tmp_path: pathlib.Path) -> None:
        """A client working in one project does not talk to a daemon serving another: it takes that daemon
        down and starts up the replacement."""
        project1 = tmp_path / 'first'
        project2 = tmp_path / 'second'
        _create_pxt_project(project1)
        _create_pxt_project(project2)

        health1 = _pxt_health(daemon_port, cwd=project1)
        assert health1['project_root'] == str(project1)
        assert psutil.pid_exists(health1['pid'])

        health2 = _pxt_health(daemon_port, cwd=project2)
        assert health2['project_root'] == str(project2)
        assert health2['pid'] != health1['pid']
        assert psutil.pid_exists(health2['pid'])
        assert not psutil.pid_exists(health1['pid'])

    def test_replace_drifted_identity_daemon(self, daemon_port: int, tmp_path: pathlib.Path) -> None:
        """A client does not talk to a daemon built from a different environment."""
        project = tmp_path / 'project'
        _create_pxt_project(project)
        drift_env_var = 'PIXELTABLE_TEST_DRIFT'

        health1 = _pxt_health(daemon_port, cwd=project, env_overrides={drift_env_var: '1'})
        assert drift_env_var in health1['pixeltable_env']
        assert psutil.pid_exists(health1['pid'])

        health2 = _pxt_health(daemon_port, cwd=project)
        assert drift_env_var not in health2['pixeltable_env']
        assert health2['pid'] != health1['pid']
        assert psutil.pid_exists(health2['pid'])
        assert not psutil.pid_exists(health1['pid'])

    @pytest.mark.skipif(sys.platform == 'win32', reason='Windows has no SIGSTOP')
    def test_replace_hung_daemon(self, daemon_port: int, tmp_path: pathlib.Path) -> None:
        project = tmp_path / 'project'
        _create_pxt_project(project)

        pid1 = _pxt_health(daemon_port, cwd=project)['pid']
        pidfile = pathlib.Path(pidfile_path(daemon_port))
        assert pidfile.read_text(encoding='utf-8').strip() == str(pid1)
        # SIGSTOP holds the current daemon. The daemon continues to hold the port but doesn't respond on it.
        os.kill(pid1, signal.SIGSTOP)
        try:
            health2 = _pxt_health(daemon_port, cwd=project)
            assert health2['project_root'] == str(project)
            assert health2['pid'] != pid1
            assert psutil.pid_exists(health2['pid'])
            assert not psutil.pid_exists(pid1)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid1, signal.SIGCONT)

    @pytest.mark.parametrize('pidfile_content', ['not-an-int', '0', '-1'])
    def test_unusable_pidfile_does_not_block_startup(
        self, daemon_port: int, tmp_path: pathlib.Path, pidfile_content: str
    ) -> None:
        project = tmp_path / 'project'
        _create_pxt_project(project)

        pathlib.Path(pidfile_path(daemon_port)).write_text(pidfile_content, encoding='utf-8')

        assert _pxt_health(daemon_port, cwd=project)['project_root'] == str(project)
