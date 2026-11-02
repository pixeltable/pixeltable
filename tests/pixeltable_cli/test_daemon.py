"""Tests for how a cli client spawns or adopts its daemon."""

import contextlib
import errno
import json
import os
import pathlib
import shutil
import site
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

import psutil
import pytest

from pixeltable_cli.server import http_server
from pixeltable_cli.utils import pidfile_path, working_dirs_path

_HEALTH_TIMEOUT_SECS = 180.0

_OLDER_RELEASE_DAEMON = pathlib.Path(__file__).with_name('older_release_daemon.py')


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


def _pxt(port: int, cwd: pathlib.Path, env_overrides: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    """Run pxt with args against the daemon on port."""
    return subprocess.run(
        ['pxt', *args],
        env={**os.environ, 'PXT_PORT': str(port), **env_overrides},
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=_HEALTH_TIMEOUT_SECS,
    )


def _replacement_pid(port: int, cwd: pathlib.Path, env_overrides: dict[str, str], pid: int) -> int:
    """Run `pxt health`, check that a new daemon replaced the one with pid, and return the new daemon's pid."""
    health = _pxt_health(port, cwd=cwd, env_overrides=env_overrides)
    assert health['pid'] != pid
    assert psutil.pid_exists(health['pid'])
    assert not psutil.pid_exists(pid)
    return health['pid']


def _user_site(userbase: pathlib.Path) -> pathlib.Path:
    """The target directory of pip install --user when PYTHONUSERBASE is userbase."""
    scheme = sysconfig.get_preferred_scheme('user')
    return pathlib.Path(sysconfig.get_path('purelib', scheme, vars={'userbase': str(userbase)}))


@contextlib.contextmanager
def _busy_daemon(port: int, project: pathlib.Path, env_overrides: dict[str, str]) -> Iterator[None]:
    """Keep the daemon on port serving a request until the block exits."""
    release = project / 'release'
    # an application file whose import waits for release
    (project / 'app.py').write_text(
        f'import pathlib, time\nwhile not pathlib.Path({str(release)!r}).exists():\n    time.sleep(0.05)\n',
        encoding='utf-8',
    )
    service_check = subprocess.Popen(
        ['pxt', 'service', 'check', 'app.py'],
        env={**os.environ, 'PXT_PORT': str(port), **env_overrides},
        cwd=project,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + _HEALTH_TIMEOUT_SECS
        while len(_pxt_health(port, cwd=project, env_overrides=env_overrides)['in_flight']) == 0:
            assert time.monotonic() < deadline, 'pxt service check never reached the daemon'
            time.sleep(0.1)
        yield
    finally:
        release.touch()
        service_check.wait(timeout=_HEALTH_TIMEOUT_SECS)


def _create_catalog_dir(port: int, project: pathlib.Path, name: str) -> None:
    """Create the catalog directory name with the example application's table in it."""
    assert _pxt(port, project, {}, 'service', 'example', '--out', 'app.py').returncode == 0
    r = _pxt(port, project, {}, 'schema', 'update', 'app.py', name)
    assert r.returncode == 0, r.stderr


def _install_metadata(path: pathlib.Path) -> None:
    """Write the metadata directory at path the way an installer does, for the distribution and version in its name."""
    name, version = path.stem.split('-')
    path.mkdir(parents=True)
    metadata_file = 'METADATA' if path.suffix == '.dist-info' else 'PKG-INFO'
    (path / metadata_file).write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n', encoding='utf-8')


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

    @pytest.mark.skipif(site.ENABLE_USER_SITE is not True, reason='a venv disables the user site directory')
    def test_replace_daemon_after_install(self, daemon_port: int, tmp_path: pathlib.Path) -> None:
        """An install, reinstall, upgrade or uninstall replaces the daemon; an unchanged install keeps it."""
        project = tmp_path / 'project'
        _create_pxt_project(project)
        userbase = tmp_path / 'userbase'
        user_env = {'PYTHONUSERBASE': str(userbase)}
        user_site = _user_site(userbase)
        v1 = user_site / 'fakepkg-1.0.dist-info'
        v2 = user_site / 'fakepkg-2.0.dist-info'
        pid = _pxt_health(daemon_port, cwd=project, env_overrides=user_env)['pid']

        _install_metadata(v1)
        pid = _replacement_pid(daemon_port, project, user_env, pid)

        # reinstall at the same version
        shutil.rmtree(v1)
        _install_metadata(v1)
        pid = _replacement_pid(daemon_port, project, user_env, pid)

        # upgrade
        shutil.rmtree(v1)
        _install_metadata(v2)
        pid = _replacement_pid(daemon_port, project, user_env, pid)

        # uninstall
        shutil.rmtree(v2)
        pid = _replacement_pid(daemon_port, project, user_env, pid)

        # a setup.py install, which writes .egg-info
        _install_metadata(user_site / 'legacypkg-1.0.egg-info')
        pid = _replacement_pid(daemon_port, project, user_env, pid)

        if sys.platform != 'win32':  # creating a symlink on Windows needs a privilege
            # a dangling metadata symlink does not restart the daemon, and does not hide an install beside it
            (user_site / 'brokenpkg-1.0.dist-info').symlink_to(tmp_path / 'gone', target_is_directory=True)
            assert _pxt_health(daemon_port, cwd=project, env_overrides=user_env)['pid'] == pid
            _install_metadata(user_site / 'fakepkg-3.0.dist-info')
            pid = _replacement_pid(daemon_port, project, user_env, pid)

        assert _pxt_health(daemon_port, cwd=project, env_overrides=user_env)['pid'] == pid

    def test_replace_older_release_daemon(self, daemon_port: int, tmp_path: pathlib.Path) -> None:
        """A daemon from an older release, which reports fewer identity fields, is replaced."""
        project = tmp_path / 'project'
        _create_pxt_project(project)
        older_daemon = subprocess.Popen(
            [sys.executable, str(_OLDER_RELEASE_DAEMON), str(daemon_port), str(project)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            assert older_daemon.stdout is not None
            assert older_daemon.stdout.readline() == 'ready\n'

            health = _pxt_health(daemon_port, cwd=project)
            older_daemon.wait(timeout=60)
            assert health['pid'] != older_daemon.pid
            assert psutil.pid_exists(health['pid'])
        finally:
            if older_daemon.poll() is None:
                older_daemon.kill()

    def test_keep_busy_mismatched_daemon(self, daemon_port: int, tmp_path: pathlib.Path) -> None:
        """A client does not replace a mismatched daemon that is serving a request, and replaces it afterwards."""
        project = tmp_path / 'project'
        _create_pxt_project(project)
        drift_env = {'PIXELTABLE_TEST_DRIFT': '1'}
        pid = _pxt_health(daemon_port, cwd=project)['pid']

        with _busy_daemon(daemon_port, project, {}):
            r = _pxt(daemon_port, project, drift_env, 'ls', '/', '--json')
            assert r.returncode == 1
            error = json.loads(r.stdout)
            assert error['error_code'] == 'DAEMON_BUSY'
            assert error['retryable'] is True
            assert 'POST /api/service/check' in error['message']
            assert 'pxt daemon restart -f' in error['message']
            assert 'pixeltable_env' not in error['message']
            assert psutil.pid_exists(pid)

        _replacement_pid(daemon_port, project, drift_env, pid)

    def test_service_without_fastapi(self, daemon_port: int, tmp_path: pathlib.Path) -> None:
        """Without fastapi, pxt schema update succeeds, and pxt service update prints the install command."""
        project = tmp_path / 'project'
        _create_pxt_project(project)
        no_fastapi = tmp_path / 'no_fastapi'
        no_fastapi.mkdir()
        (no_fastapi / 'fastapi.py').write_text(
            "raise ModuleNotFoundError(\"No module named 'fastapi'\", name='fastapi')\n", encoding='utf-8'
        )
        env = {'PYTHONPATH': str(no_fastapi)}

        assert _pxt(daemon_port, project, env, 'service', 'example', '--out', 'app.py').returncode == 0
        r = _pxt(daemon_port, project, env, 'schema', 'update', 'app.py', 'no_fastapi')
        assert r.returncode == 0, r.stderr

        r = _pxt(daemon_port, project, env, 'service', 'update', '-f', 'app.py', 'no_fastapi')
        assert r.returncode == 1
        assert "pip install 'pixeltable[serve]'" in r.stderr
        assert '[UNSUPPORTED_OPERATION]' in r.stderr
        assert 'Traceback' not in r.stderr

    def test_restart_keeps_working_dir(self, daemon_port: int, tmp_path: pathlib.Path) -> None:
        """A replaced or restarted daemon keeps the working directory set with pxt cd.

        pxt daemon stop and a daemon for another project clear it.
        """
        project = tmp_path / 'project'
        _create_pxt_project(project)
        other_project = tmp_path / 'other_project'
        _create_pxt_project(other_project)
        drift_env = {'PIXELTABLE_TEST_DRIFT': '1'}
        _create_catalog_dir(daemon_port, project, 'restart_wd')
        r = _pxt(daemon_port, project, {}, 'cd', 'restart_wd')
        assert r.returncode == 0, r.stderr
        pid = _pxt_health(daemon_port, cwd=project)['pid']

        _replacement_pid(daemon_port, project, drift_env, pid)
        assert 'restart_wd' in _pxt(daemon_port, project, drift_env, 'pwd').stdout

        assert _pxt(daemon_port, project, drift_env, 'daemon', 'restart').returncode == 0
        assert 'restart_wd' in _pxt(daemon_port, project, drift_env, 'pwd').stdout

        assert _pxt(daemon_port, project, drift_env, 'daemon', 'stop').returncode == 0
        assert _pxt(daemon_port, project, drift_env, 'pwd').stdout == '(no working directory)\n'

        r = _pxt(daemon_port, project, drift_env, 'cd', 'restart_wd')
        assert r.returncode == 0, r.stderr
        assert _pxt(daemon_port, other_project, drift_env, 'pwd').stdout == '(no working directory)\n'
        assert _pxt(daemon_port, project, drift_env, 'pwd').stdout == '(no working directory)\n'

    def test_replace_hung_daemon(self, daemon_port: int, tmp_path: pathlib.Path) -> None:
        project = tmp_path / 'project'
        _create_pxt_project(project)

        pid1 = _pxt_health(daemon_port, cwd=project)['pid']
        pidfile = pathlib.Path(pidfile_path(daemon_port))
        assert pidfile.read_text(encoding='utf-8').strip() == str(pid1)
        # Suspending the current daemon holds it: it continues to hold the port but doesn't respond on it.
        daemon = psutil.Process(pid1)
        daemon.suspend()
        try:
            health2 = _pxt_health(daemon_port, cwd=project)
            assert health2['project_root'] == str(project)
            assert health2['pid'] != pid1
            assert psutil.pid_exists(health2['pid'])
            assert not psutil.pid_exists(pid1)
        finally:
            with contextlib.suppress(psutil.NoSuchProcess):
                daemon.resume()

    def test_keep_briefly_unresponsive_daemon(self, daemon_port: int, tmp_path: pathlib.Path) -> None:
        """A daemon that misses a health probe but answers within the client's grace period keeps serving, and
        keeps what it holds for its clients, such as their working directories."""
        project = tmp_path / 'project'
        _create_pxt_project(project)

        pid1 = _pxt_health(daemon_port, cwd=project)['pid']
        daemon = psutil.Process(pid1)
        daemon.suspend()
        try:
            client = subprocess.Popen(
                ['pxt', 'health'],
                env={**os.environ, 'PXT_PORT': str(daemon_port)},
                cwd=project,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # long enough for the client to start up and miss its first probe, well inside its grace period
            time.sleep(5.0)
        finally:
            with contextlib.suppress(psutil.NoSuchProcess):
                daemon.resume()
        out, err = client.communicate(timeout=_HEALTH_TIMEOUT_SECS)

        assert client.returncode == 0, err
        assert json.loads(out)['pid'] == pid1

    def test_second_daemon_defers(self, daemon_port: int, tmp_path: pathlib.Path) -> None:
        """A daemon started on a port that another daemon serves leaves the port to that daemon."""
        project = tmp_path / 'project'
        _create_pxt_project(project)

        pid1 = _pxt_health(daemon_port, cwd=project)['pid']
        # launched the way a client launches a daemon, which then finds the port taken and exits rather than serve
        subprocess.run(
            [sys.executable, '-m', 'pixeltable_cli.server.daemon', '--project-root', str(project)],
            env={**os.environ, 'PXT_PORT': str(daemon_port)},
            cwd=project,
            capture_output=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=_HEALTH_TIMEOUT_SECS,
        )

        assert pathlib.Path(pidfile_path(daemon_port)).read_text(encoding='utf-8').strip() == str(pid1)
        assert _pxt_health(daemon_port, cwd=project)['pid'] == pid1

    @pytest.mark.parametrize('pidfile_content', ['not-an-int', '0', '-1'])
    def test_unusable_pidfile_does_not_block_startup(
        self, daemon_port: int, tmp_path: pathlib.Path, pidfile_content: str
    ) -> None:
        project = tmp_path / 'project'
        _create_pxt_project(project)

        pathlib.Path(pidfile_path(daemon_port)).write_text(pidfile_content, encoding='utf-8')

        assert _pxt_health(daemon_port, cwd=project)['project_root'] == str(project)

    @pytest.mark.parametrize('content', ['not json', '{"working_dirs": []}'])
    def test_unusable_working_dirs_file(self, daemon_port: int, tmp_path: pathlib.Path, content: str) -> None:
        """The daemon starts without working directories when their file is unusable."""
        project = tmp_path / 'project'
        _create_pxt_project(project)

        pathlib.Path(working_dirs_path(daemon_port)).write_text(content, encoding='utf-8')

        assert _pxt(daemon_port, project, {}, 'pwd').stdout == '(no working directory)\n'

    def test_unwritable_working_dirs_file(self, daemon_port: int, tmp_path: pathlib.Path) -> None:
        """The daemon starts, and pxt cd works, when the working directories' file cannot be written."""
        project = tmp_path / 'project'
        _create_pxt_project(project)
        working_dirs_file = pathlib.Path(working_dirs_path(daemon_port))
        working_dirs_file.mkdir()
        try:
            _create_catalog_dir(daemon_port, project, 'unwritable_wd')
            r = _pxt(daemon_port, project, {}, 'cd', 'unwritable_wd')
            assert r.returncode == 0, r.stderr
            assert 'unwritable_wd' in _pxt(daemon_port, project, {}, 'pwd').stdout

            r = _pxt(daemon_port, project, {}, 'cd')
            assert r.returncode == 0, r.stderr
            assert _pxt(daemon_port, project, {}, 'pwd').stdout == '(no working directory)\n'
        finally:
            shutil.rmtree(working_dirs_file)


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
