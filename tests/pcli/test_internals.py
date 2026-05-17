"""Unit tests for pcli internals.

Covers things that aren't reachable through the daemon smoke tests:
  - probe.py spawn / restart / kill safety paths (monkeypatched)
  - the confirm.py interactive prompt
  - parser.py / main.py error and help paths
  - http.py client error branches
  - the interactive shell REPL (driven via subprocess.Popen)
"""

import importlib.metadata
import io
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import urllib.error
from collections.abc import Iterator
from email.message import Message

import pytest
from typing_extensions import Self

# The daemon requires the `cli` extra; skip the whole module on `minimal` installs.
pytest.importorskip('fastapi')
pytest.importorskip('uvicorn')

from pcli import probe
from pcli.client import confirm, http, main as client_main, parser as client_parser
from pcli.client.commands import shell as shell_cmd, status as status_cmd
from pcli.server import daemon as server_daemon, routes as server_routes
from pixeltable import exceptions as excs


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def fresh_port(init_env: None) -> Iterator[int]:
    """Allocate a port no daemon is using, and tear down any daemon left running on it."""
    port = _pick_port()
    prior = os.environ.get('PCLI_PORT')
    os.environ['PCLI_PORT'] = str(port)
    try:
        yield port
    finally:
        pid = probe._read_pidfile()
        if pid is not None:
            # Reuse the production kill helper: it already handles the SIGKILL fallback
            # and the Windows quirks around os.kill(pid, 0). Cleanup is best-effort.
            try:
                probe._kill_and_wait(pid, timeout=3.0)
            except Exception:
                pass
        if prior is None:
            os.environ.pop('PCLI_PORT', None)
        else:
            os.environ['PCLI_PORT'] = prior


class TestProbe:
    """Spawn / restart / kill safety paths."""

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
        killed: list[int | str] = []
        monkeypatch.setattr(probe, '_kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        monkeypatch.setattr(probe, 'spawn_detached', lambda: killed.append('spawn'))

        with pytest.raises(RuntimeError, match='does not match our pidfile'):
            probe.ensure_running()
        assert killed == []

    def test_version_mismatch_restart_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Matching pidfile + version mismatch: ensure_running kills the old daemon, spawns
        a new one, and cross-verifies the post-restart responder reports our version."""
        responses = iter(
            [
                {'ok': True, 'service': 'pcli', 'pxt_version': 'OLD', 'pid': 100, 'started_at': 'a'},
                {'ok': True, 'service': 'pcli', 'pxt_version': 'NEW', 'pid': 200, 'started_at': 'b'},
            ]
        )
        monkeypatch.setattr(probe, '_fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(probe, '_client_pxt_version', lambda: 'NEW')
        monkeypatch.setattr(probe, '_read_pidfile', lambda: 100)
        actions: list[tuple[str, ...] | tuple[str, int]] = []
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

    def test_pidfile_malformed(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(probe, 'pidfile_path', lambda: str(tmp_path / 'bogus.pid'))
        with open(probe.pidfile_path(), 'w', encoding='utf-8') as f:
            f.write('not-an-int')
        assert probe._read_pidfile() is None

    def test_pidfile_missing(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(probe, 'pidfile_path', lambda: str(tmp_path / 'missing.pid'))
        assert probe._read_pidfile() is None

    def test_fetch_health_rejects_non_pcli_marker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A responder that returns {ok: true} but isn't us is not our daemon."""

        class FakeResp:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *a: object) -> None:
                pass

            def read(self) -> bytes:
                return self._body

        monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: FakeResp(b'{"ok": true, "service": "other"}'))
        assert probe._fetch_health() is None

    def test_fetch_health_missing_identity_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeResp:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *a: object) -> None:
                pass

            def read(self) -> bytes:
                return b'{"ok": true, "service": "pcli"}'

        monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: FakeResp())
        # missing pxt_version/pid/started_at -> rejected
        assert probe._fetch_health() is None

    def test_fetch_health_url_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*a: object, **kw: object) -> None:
            raise urllib.error.URLError('refused')

        monkeypatch.setattr('urllib.request.urlopen', boom)
        assert probe._fetch_health() is None

    def test_client_pxt_version_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(probe.importlib.metadata, 'version', boom)
        assert probe._client_pxt_version() is None

    def test_check_daemon_deps_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(probe.importlib.util, 'find_spec', lambda name: None)
        with pytest.raises(RuntimeError, match=r"pip install 'pixeltable\[cli\]'"):
            probe._check_daemon_deps()

    def test_spawn_detached_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(probe, '_check_daemon_deps', lambda: None)

        def boom(*a: object, **kw: object) -> None:
            raise OSError('disk full')

        monkeypatch.setattr(probe.os, 'makedirs', boom)
        with pytest.raises(RuntimeError, match='pcli daemon log unavailable'):
            probe.spawn_detached()

    def test_tail_daemon_log_missing(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        # log file does not exist -> empty string, no exception
        assert probe._tail_daemon_log() == ''

    def test_tail_daemon_log_truncates(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        log_path = probe._daemon_log_path()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            for i in range(50):
                f.write(f'line {i}\n')
        tail = probe._tail_daemon_log(n_lines=3)
        assert tail.splitlines() == ['line 47', 'line 48', 'line 49']

    def test_wait_for_health_timeout_includes_log_tail(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        log_path = probe._daemon_log_path()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('startup blew up: address already in use\n')
        monkeypatch.setattr(probe, 'is_running', lambda timeout=0.3: False)
        with pytest.raises(RuntimeError, match='did not come up') as ei:
            probe.wait_for_health(timeout=0.2)
        assert 'address already in use' in str(ei.value)

    def test_kill_and_wait_falls_through_to_sigkill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If SIGTERM doesn't bring the daemon down, _kill_and_wait must follow up with SIGKILL.

        Liveness is checked via os.kill(pid, 0), not /health, so a hung-but-alive daemon
        still holding the listen socket gets SIGKILLed instead of leaving the socket bound.
        """
        calls: list[int] = []

        def fake_kill(pid: int, sig: int) -> None:
            # never raises -> _pid_alive returns True every iteration, deadline expires
            calls.append(sig)

        monkeypatch.setattr(probe.os, 'kill', fake_kill)
        probe._kill_and_wait(12345, timeout=0.2)
        assert signal.SIGTERM in calls
        # On non-Windows we have a real SIGKILL; on Windows it falls back to SIGTERM.
        assert getattr(signal, 'SIGKILL', signal.SIGTERM) in calls

    def test_kill_and_wait_returns_when_pid_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIGTERM goes through, then the PID exits -> return without escalating to SIGKILL."""
        calls: list[int] = []

        def fake_kill(pid: int, sig: int) -> None:
            calls.append(sig)
            # signal 0 is the liveness probe: raise to simulate the PID being gone
            if sig == 0:
                raise ProcessLookupError

        monkeypatch.setattr(probe.os, 'kill', fake_kill)
        probe._kill_and_wait(12345, timeout=1.0)
        sigkill = getattr(signal, 'SIGKILL', signal.SIGTERM)
        assert signal.SIGTERM in calls
        # On platforms where SIGKILL != SIGTERM, it must NOT have been issued.
        if sigkill != signal.SIGTERM:
            assert sigkill not in calls

    def test_kill_and_wait_no_such_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(pid: int, sig: int) -> None:
            raise ProcessLookupError

        monkeypatch.setattr(probe.os, 'kill', boom)
        # Should return cleanly without raising
        probe._kill_and_wait(99999)

    def test_pid_alive_dead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(pid: int, sig: int) -> None:
            raise ProcessLookupError

        monkeypatch.setattr(probe.os, 'kill', boom)
        assert probe._pid_alive(99999) is False

    def test_pid_alive_alive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(probe.os, 'kill', lambda pid, sig: None)
        assert probe._pid_alive(12345) is True

    def test_pid_alive_permission_denied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PermissionError means the PID exists but is owned by another user; treat as alive."""

        def boom(pid: int, sig: int) -> None:
            raise PermissionError

        monkeypatch.setattr(probe.os, 'kill', boom)
        assert probe._pid_alive(1) is True

    def test_pid_alive_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(pid: int, sig: int) -> None:
            raise OSError('einval')

        monkeypatch.setattr(probe.os, 'kill', boom)
        assert probe._pid_alive(0) is False


class TestConfirm:
    def test_force_short_circuits(self) -> None:
        # No TTY, no input - force=True must just return.
        confirm.confirm_or_exit('drop something?', force=True)

    def test_no_tty_refuses(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(confirm, '_stdin_is_real_tty', lambda: False)
        with pytest.raises(SystemExit) as ei:
            confirm.confirm_or_exit('drop something?', force=False)
        assert ei.value.code == 2
        assert '--force' in capsys.readouterr().err

    def test_tty_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(confirm, '_stdin_is_real_tty', lambda: True)
        monkeypatch.setattr(confirm.sys, 'stdin', io.StringIO('y\n'))
        # Should not raise.
        confirm.confirm_or_exit('drop something?', force=False)

    def test_tty_no(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(confirm, '_stdin_is_real_tty', lambda: True)
        monkeypatch.setattr(confirm.sys, 'stdin', io.StringIO('n\n'))
        with pytest.raises(SystemExit) as ei:
            confirm.confirm_or_exit('drop something?', force=False)
        assert ei.value.code == 1
        assert 'aborted' in capsys.readouterr().err

    def test_tty_empty_aborts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(confirm, '_stdin_is_real_tty', lambda: True)
        monkeypatch.setattr(confirm.sys, 'stdin', io.StringIO('\n'))
        with pytest.raises(SystemExit):
            confirm.confirm_or_exit('drop something?', force=False)

    def test_stdin_is_real_tty_posix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-Windows path: isatty() True -> returns True without touching ctypes."""

        class FakeStdin:
            def isatty(self) -> bool:
                return True

        monkeypatch.setattr(confirm.sys, 'stdin', FakeStdin())
        monkeypatch.setattr(confirm.sys, 'platform', 'linux')
        assert confirm._stdin_is_real_tty() is True

    def test_stdin_is_real_tty_not_a_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeStdin:
            def isatty(self) -> bool:
                return False

        monkeypatch.setattr(confirm.sys, 'stdin', FakeStdin())
        assert confirm._stdin_is_real_tty() is False


class TestParser:
    def test_error_exits_with_epilog(self, capsys: pytest.CaptureFixture) -> None:
        p = client_parser.Parser(prog='pcli foo', epilog='Examples:\n  pcli foo bar')
        p.add_argument('required')
        with pytest.raises(SystemExit) as ei:
            p.parse_args([])
        assert ei.value.code == 2
        err = capsys.readouterr().err
        assert 'pcli foo' in err
        assert 'Examples:' in err

    def test_parse_cols_none(self) -> None:
        p = client_parser.Parser(prog='pcli x')
        assert client_parser.parse_cols(None, p) is None

    def test_parse_cols_valid(self) -> None:
        p = client_parser.Parser(prog='pcli x')
        assert client_parser.parse_cols('a,b, c', p) == ['a', 'b', 'c']

    @pytest.mark.parametrize('arg', ['a,', ',a', 'a,,b', ',', '  ,a'])
    def test_parse_cols_rejects_empty_tokens(self, arg: str, capsys: pytest.CaptureFixture) -> None:
        p = client_parser.Parser(prog='pcli x')
        with pytest.raises(SystemExit) as ei:
            client_parser.parse_cols(arg, p)
        assert ei.value.code == 2
        assert 'must not be empty' in capsys.readouterr().err


class TestMain:
    def test_print_help_lists_every_command(self, capsys: pytest.CaptureFixture) -> None:
        client_main._print_help()
        out = capsys.readouterr().out
        assert all(cmd in out for cmd in client_main.COMMANDS)

    def test_dispatch_unknown_command(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as ei:
            client_main.dispatch('not_a_real_cmd', [])
        assert ei.value.code == 2
        assert 'unknown command' in capsys.readouterr().err

    def test_main_help_flag(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(client_main.sys, 'argv', ['pcli', '--help'])
        with pytest.raises(SystemExit) as ei:
            client_main.main()
        assert ei.value.code == 0
        assert 'commands:' in capsys.readouterr().out

    def test_main_no_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(client_main.sys, 'argv', ['pcli'])
        with pytest.raises(SystemExit) as ei:
            client_main.main()
        assert ei.value.code == 2


class TestHttp:
    def test_ensure_running_failure(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        def boom() -> str:
            raise RuntimeError('cannot spawn daemon: missing fastapi')

        monkeypatch.setattr(http, 'ensure_running', boom)
        with pytest.raises(SystemExit) as ei:
            http.get('/pcli/v0/health')
        assert ei.value.code == 1
        assert 'cannot spawn daemon' in capsys.readouterr().err

    def test_http_error_with_detail(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(http, 'ensure_running', lambda: 'http://127.0.0.1:1')

        def raise_http(*a: object, **kw: object) -> None:
            body = io.BytesIO(b'{"detail": "n must be > 0"}')
            raise urllib.error.HTTPError('http://x', 400, 'Bad Request', Message(), body)

        monkeypatch.setattr(http.urllib.request, 'urlopen', raise_http)
        with pytest.raises(SystemExit) as ei:
            http.post('/pcli/v0/rows', {'path': 't', 'n': 0, 'cols': None})
        assert ei.value.code == 1
        err = capsys.readouterr().err
        assert '400' in err
        assert 'n must be > 0' in err

    def test_http_error_unparseable_body(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(http, 'ensure_running', lambda: 'http://127.0.0.1:1')

        def raise_http(*a: object, **kw: object) -> None:
            raise urllib.error.HTTPError(
                'http://x', 500, 'Internal Server Error', Message(), io.BytesIO(b'<html>not json</html>')
            )

        monkeypatch.setattr(http.urllib.request, 'urlopen', raise_http)
        with pytest.raises(SystemExit) as ei:
            http.get('/pcli/v0/health')
        assert ei.value.code == 1
        err = capsys.readouterr().err
        # falls back to e.reason when the body isn't JSON
        assert 'Internal Server Error' in err

    def test_url_error_unreachable(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(http, 'ensure_running', lambda: 'http://127.0.0.1:1')

        def boom(*a: object, **kw: object) -> None:
            raise urllib.error.URLError('connection refused')

        monkeypatch.setattr(http.urllib.request, 'urlopen', boom)
        with pytest.raises(SystemExit) as ei:
            http.get('/pcli/v0/health')
        assert ei.value.code == 1
        assert 'cannot reach daemon' in capsys.readouterr().err


class TestShell:
    """Exercise the REPL via subprocess to cover input/eof/error branches."""

    def test_shell_runs_health_then_exits(self, pcli_daemon: int) -> None:
        env = {**os.environ, 'PCLI_PORT': str(pcli_daemon)}
        r = subprocess.run(
            ['pcli', 'shell'], input='health\nexit\n', capture_output=True, text=True, env=env, timeout=30, check=False
        )
        assert r.returncode == 0, r.stderr
        # the health response is JSON; should appear in stdout between two prompts
        assert '"service": "pcli"' in r.stdout

    def test_shell_eof_exits_cleanly(self, pcli_daemon: int) -> None:
        env = {**os.environ, 'PCLI_PORT': str(pcli_daemon)}
        r = subprocess.run(
            ['pcli', 'shell'],
            input='',  # immediate EOF
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        assert r.returncode == 0

    def test_shell_unknown_command_does_not_kill_session(self, pcli_daemon: int) -> None:
        env = {**os.environ, 'PCLI_PORT': str(pcli_daemon)}
        r = subprocess.run(
            ['pcli', 'shell'],
            input='not_a_cmd\nhealth\nexit\n',
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        assert r.returncode == 0
        # bad command produces a stderr line, but the follow-up `health` still runs
        assert 'unknown command' in r.stderr
        assert '"service": "pcli"' in r.stdout

    def test_shell_rejects_nested_shell(self, pcli_daemon: int) -> None:
        env = {**os.environ, 'PCLI_PORT': str(pcli_daemon)}
        r = subprocess.run(
            ['pcli', 'shell'], input='shell\nexit\n', capture_output=True, text=True, env=env, timeout=30, check=False
        )
        assert r.returncode == 0
        assert 'already in shell' in r.stderr

    def test_shell_help(self, pcli_daemon: int) -> None:
        env = {**os.environ, 'PCLI_PORT': str(pcli_daemon)}
        r = subprocess.run(
            ['pcli', 'shell'], input='help\nexit\n', capture_output=True, text=True, env=env, timeout=30, check=False
        )
        assert r.returncode == 0
        # `help` lists every non-shell command
        assert all(c in r.stdout for c in ('health', 'ls', 'describe'))

    def test_shell_empty_line(self, pcli_daemon: int) -> None:
        env = {**os.environ, 'PCLI_PORT': str(pcli_daemon)}
        r = subprocess.run(
            ['pcli', 'shell'],
            input='\n\nhealth\nexit\n',
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        assert r.returncode == 0
        assert '"service": "pcli"' in r.stdout

    def test_shell_parse_error(self, pcli_daemon: int) -> None:
        env = {**os.environ, 'PCLI_PORT': str(pcli_daemon)}
        # unterminated quote -> shlex.split raises ValueError
        r = subprocess.run(
            ['pcli', 'shell'],
            input='ls "unterminated\nexit\n',
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        assert r.returncode == 0
        assert 'parse error' in r.stderr

    def test_shell_help_prints_every_command(self, capsys: pytest.CaptureFixture) -> None:
        shell_cmd._print_help(client_main.COMMANDS)
        out = capsys.readouterr().out
        # shell suppresses its own entry; every other command appears
        assert 'shell' not in [line.split()[0] for line in out.splitlines() if line.strip() != '']
        assert all(cmd in out for cmd in client_main.COMMANDS if cmd != 'shell')


class TestStatusFmtSize:
    @pytest.mark.parametrize(
        'n,expected_suffix',
        [(None, '-'), (0, 'B'), (2048, 'KB'), (3 * 1024**2, 'MB'), (5 * 1024**4, 'TB'), (10 * 1024**5, 'PB')],
    )
    def test_fmt_size(self, n: int | None, expected_suffix: str) -> None:
        out = status_cmd._fmt_size(n)
        assert expected_suffix in out


class TestServerDaemon:
    def test_pidfile_lifecycle(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = str(tmp_path / 'sub' / 'pid')
        monkeypatch.setattr(server_daemon, 'pidfile_path', lambda: path)
        server_daemon._write_pidfile()
        with open(path, encoding='utf-8') as f:
            assert int(f.read().strip()) == os.getpid()
        server_daemon._remove_pidfile()
        assert not os.path.exists(path)

    def test_remove_pidfile_missing_no_raise(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server_daemon, 'pidfile_path', lambda: str(tmp_path / 'never-existed'))
        # Must not raise even if the file isn't there.
        server_daemon._remove_pidfile()


class TestServerRouteHelpers:
    """Cover routes.py helpers reachable without spinning up the daemon."""

    def test_dir_size_none(self) -> None:
        assert server_routes._dir_size(None) is None

    def test_dir_size_missing(self, tmp_path: pathlib.Path) -> None:
        assert server_routes._dir_size(str(tmp_path / 'nope')) is None

    def test_dir_size_sums_files(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / 'a').write_bytes(b'x' * 10)
        sub = tmp_path / 'sub'
        sub.mkdir()
        (sub / 'b').write_bytes(b'y' * 5)
        assert server_routes._dir_size(str(tmp_path)) == 15

    def test_dir_size_skips_stat_errors(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / 'a').write_bytes(b'x' * 10)
        real_stat = os.stat

        def flaky(p: str) -> object:
            if p.endswith('a'):
                raise OSError('vanished')
            return real_stat(p)

        monkeypatch.setattr(server_routes.os, 'stat', flaky)
        # the failing file is skipped; the walk still completes
        assert server_routes._dir_size(str(tmp_path)) == 0

    def test_redact_db_password_none(self) -> None:
        assert server_routes._redact_db_password(None) is None

    def test_redact_db_password_hides_password(self) -> None:
        out = server_routes._redact_db_password('postgresql://user:secret@host/db')
        assert out is not None
        assert 'secret' not in out

    def test_redact_db_password_unparseable(self) -> None:
        # malformed URL -> caught and returns None rather than 500ing /status
        assert server_routes._redact_db_password('::: not a url :::') is None

    def test_safe_count_swallows_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeT:
            def count(self) -> int:
                raise excs.NotFoundError(excs.ErrorCode.PATH_NOT_FOUND, 'catalog gone')

        monkeypatch.setattr(server_routes.pxt, 'get_table', lambda p: FakeT())
        assert server_routes._tbl_count('any/path') is None
