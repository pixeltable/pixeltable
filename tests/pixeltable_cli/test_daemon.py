"""Tests for how a cli client spawns or adopts its daemon."""

import contextlib
import errno
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from typing import Any

import psutil
import pytest

from pixeltable_cli.server import http_server
from pixeltable_cli.utils import pidfile_path

_HEALTH_TIMEOUT_SECS = 180.0


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


def _request_daemon(
    port: int,
    path: str,
    *,
    host: str,
    body: bytes | None = None,
    content_type: str | None = None,
    address: str = '127.0.0.1',
) -> tuple[int, dict[str, Any]]:
    """Send a request to the daemon on address:port with the given Host header, and return the status and the answer."""
    headers = {'Host': host}
    if content_type is not None:
        headers['Content-Type'] = content_type
    req = urllib.request.Request(f'http://{address}:{port}{path}', data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestBrowserRequests:
    """Any web page in a browser on this machine can send requests to the daemon's loopback port."""

    @pytest.mark.parametrize(
        ('host', 'status'), [('attacker.example:{port}', 403), ('127.0.0.1:{port}', 200), ('LOCALHOST:{port}', 200)]
    )
    def test_host_header(self, pxt_daemon: int, host: str, status: int) -> None:
        """A page that rebinds its own host name to 127.0.0.1 reaches the port, but sends that name as Host."""
        code, answer = _request_daemon(pxt_daemon, '/api/health', host=host.format(port=pxt_daemon))

        assert code == status
        if status == 403:
            assert 'only answers requests addressed to' in answer['detail']

    def test_host_header_other_loopback_address(self) -> None:
        """A daemon bound to another loopback address answers requests addressed to that address."""
        try:
            server = http_server.bind('127.0.0.2', 0)
        except OSError as e:
            if e.errno != errno.EADDRNOTAVAIL:
                raise
            pytest.skip('127.0.0.2 is not a local address here, as on macOS, which assigns 127.0.0.1 alone')
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            own_code, _ = _request_daemon(port, '/api/health', host=f'127.0.0.2:{port}', address='127.0.0.2')
            foreign_code, answer = _request_daemon(
                port, '/api/health', host=f'attacker.example:{port}', address='127.0.0.2'
            )
        finally:
            server.shutdown()
            server.server_close()

        assert own_code == 200
        assert foreign_code == 403
        assert answer['detail'] == f'this daemon only answers requests addressed to 127.0.0.2:{port}'

    def test_form_post(self, pxt_daemon: int) -> None:
        """A page needs no CORS preflight to post a form, but it does to post JSON."""
        code, answer = _request_daemon(
            pxt_daemon,
            '/api/cwd',
            host=f'127.0.0.1:{pxt_daemon}',
            body=b'uri=elsewhere',
            content_type='application/x-www-form-urlencoded',
        )

        assert code == 415
        assert 'application/json' in answer['detail']

    @pytest.mark.parametrize('spelling', ['LOCALHOST', 'LocalHost'])
    def test_bind_loopback_hostname_spellings(self, spelling: str) -> None:
        """A daemon bound to loopback by any spelling of localhost refuses what a web page could forge."""
        server = http_server.bind(spelling, 0)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            own_code, _ = _request_daemon(port, '/api/health', host=f'{spelling}:{port}')
            foreign_code, answer = _request_daemon(port, '/api/health', host=f'attacker.example:{port}')
            form_code, _ = _request_daemon(
                port,
                '/api/cwd',
                host=f'localhost:{port}',
                body=b'uri=elsewhere',
                content_type='application/x-www-form-urlencoded',
            )
        finally:
            server.shutdown()
            server.server_close()

        assert own_code == 200
        assert foreign_code == 403
        assert (
            answer['detail'] == f'this daemon only answers requests addressed to 127.0.0.1:{port} or localhost:{port}'
        )
        assert form_code == 415

    def test_bind_beyond_loopback(self) -> None:
        """A daemon bound beyond loopback, as on a hosted pod behind the gateway, answers any Host."""
        server = http_server.bind('0.0.0.0', 0)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            code, _ = _request_daemon(port, '/api/health', host=f'attacker.example:{port}')
        finally:
            server.shutdown()
            server.server_close()

        assert code == 200
