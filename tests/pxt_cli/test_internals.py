"""Unit tests for cli internals.

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

from pixeltable import exceptions as excs
from pxt_cli import probe
from pxt_cli.client import confirm, http, main as client_main, parser as client_parser
from pxt_cli.client.commands import daemon as daemon_cmd, shell as shell_cmd, status as status_cmd
from pxt_cli.server import daemon as server_daemon, routes as server_routes


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


# A canonical identity dict used by the probe tests below. Real values are too tied to the
# host environment to assert against; tests pin the dict via _patch_identity and pass
# matching responses (or override one field to provoke a mismatch).
_DEFAULT_IDENTITY: dict[str, object] = {
    'pxt_version': 'NEW',
    'pxt_install_dir': '/opt/site-packages/pixeltable',
    'python_executable': '/opt/conda/envs/pxt/bin/python',
    'pixeltable_home': '/home/u/.pixeltable',
    'pixeltable_pgdata': '/home/u/.pixeltable/pgdata',
    'pixeltable_config_file': '/home/u/.pixeltable/config.toml',
    'pixeltable_env': {},
}


def _patch_identity(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]) -> dict[str, object]:
    """Pin probe.identity() to a known dict so tests don't depend on the host environment."""
    ident = {**_DEFAULT_IDENTITY, **overrides}
    monkeypatch.setattr(probe, 'identity', lambda: dict(ident))
    return ident


def _health_payload(*, pid: int = 100, started_at: str = 'a', **identity_overrides: object) -> dict[str, object]:
    """Build a /health response dict shaped like the real daemon's, with identity fields
    matching _DEFAULT_IDENTITY by default. Override any field to simulate drift."""
    body: dict[str, object] = {
        'ok': True,
        'service': 'pxt',
        'pid': pid,
        'started_at': started_at,
        **_DEFAULT_IDENTITY,
        **identity_overrides,
    }
    return body


@pytest.fixture
def fresh_port(init_env: None) -> Iterator[int]:
    """Allocate a port no daemon is using, and tear down any daemon left running on it."""
    port = _pick_port()
    prior = os.environ.get('PXT_PORT')
    os.environ['PXT_PORT'] = str(port)
    try:
        yield port
    finally:
        pid = probe.read_pidfile()
        if pid is not None:
            # Reuse the production kill helper: it already handles the SIGKILL fallback
            # and the Windows quirks around os.kill(pid, 0). Cleanup is best-effort.
            try:
                probe.kill_and_wait(pid, timeout=3.0)
            except Exception:
                pass
        if prior is None:
            os.environ.pop('PXT_PORT', None)
        else:
            os.environ['PXT_PORT'] = prior


class TestProbe:
    """Spawn / restart / kill safety paths."""

    def test_auto_spawn_when_no_daemon_running(self, fresh_port: int) -> None:
        """Cold start: no daemon on the port, the cli client spawns one and routes the command."""
        env = {**os.environ, 'PXT_PORT': str(fresh_port)}
        r = subprocess.run(
            [sys.executable, '-m', 'pxt_cli.client.main', 'health'],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=30,
        )
        assert r.returncode == 0, f'pxt health failed (rc={r.returncode}): {r.stderr}'
        body = json.loads(r.stdout)
        assert body['service'] == 'pxt'
        assert body['ok'] is True
        assert body['pid'] > 0
        # the spawned daemon's pidfile should now exist and contain that PID
        assert probe.read_pidfile() == body['pid']

    def test_identity_mismatch_refuses_unknown_pid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Safety: if a responder reports a different identity AND its PID doesn't match
        our pidfile, refuse to SIGTERM it. It might be an unrelated process on the same port."""
        _patch_identity(monkeypatch, {'pxt_version': 'NEW'})
        foreign_responder = _health_payload(pxt_version='OLD', pid=99999)
        monkeypatch.setattr(probe, 'fetch_health', lambda *a, **kw: foreign_responder)
        monkeypatch.setattr(probe, 'read_pidfile', lambda: 12345)
        killed: list[int | str] = []
        monkeypatch.setattr(probe, 'kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        monkeypatch.setattr(probe, 'spawn_detached', lambda: killed.append('spawn'))

        with pytest.raises(RuntimeError, match='does not match our pidfile'):
            probe.ensure_running()
        assert killed == []

    def test_identity_mismatch_restart_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Matching pidfile + identity drift: ensure_running kills the old daemon, spawns a
        new one, and cross-verifies the post-restart responder's identity matches ours."""
        _patch_identity(monkeypatch, {'pxt_version': 'NEW'})
        responses = iter([_health_payload(pxt_version='OLD', pid=100), _health_payload(pxt_version='NEW', pid=200)])
        monkeypatch.setattr(probe, 'fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(probe, 'read_pidfile', lambda: 100)
        actions: list[tuple[str, ...] | tuple[str, int]] = []
        monkeypatch.setattr(probe, 'kill_and_wait', lambda pid, timeout=5.0: actions.append(('kill', pid)))
        monkeypatch.setattr(probe, 'spawn_detached', lambda: actions.append(('spawn',)))
        monkeypatch.setattr(probe, 'wait_for_health', lambda timeout=15.0: None)

        url = probe.ensure_running()
        assert url.startswith('http://127.0.0.1:')
        assert actions == [('kill', 100), ('spawn',)]

    def test_cross_verify_kept_killed_pid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Post-restart cross-verify: the new responder still reports the killed PID."""
        _patch_identity(monkeypatch, {'pxt_version': 'NEW'})
        responses = iter([_health_payload(pxt_version='OLD', pid=100), _health_payload(pxt_version='NEW', pid=100)])
        monkeypatch.setattr(probe, 'fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(probe, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(probe, 'kill_and_wait', lambda pid, timeout=5.0: None)
        monkeypatch.setattr(probe, 'spawn_detached', lambda: None)
        monkeypatch.setattr(probe, 'wait_for_health', lambda timeout=15.0: None)

        with pytest.raises(RuntimeError, match='new daemon kept the killed PID 100'):
            probe.ensure_running()

    def test_cross_verify_no_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Post-restart cross-verify: the new daemon never responds to /health."""
        _patch_identity(monkeypatch, {'pxt_version': 'NEW'})
        responses = iter([_health_payload(pxt_version='OLD', pid=100), None])
        monkeypatch.setattr(probe, 'fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(probe, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(probe, 'kill_and_wait', lambda pid, timeout=5.0: None)
        monkeypatch.setattr(probe, 'spawn_detached', lambda: None)
        monkeypatch.setattr(probe, 'wait_for_health', lambda timeout=15.0: None)

        with pytest.raises(RuntimeError, match='new daemon did not respond'):
            probe.ensure_running()

    def test_cross_verify_identity_still_differs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Post-restart cross-verify: a fresh PID came up but still has the wrong identity."""
        _patch_identity(monkeypatch, {'pxt_version': 'NEW'})
        responses = iter(
            [
                _health_payload(pxt_version='OLD', pid=100),
                _health_payload(pxt_version='OLD', pid=200),  # fresh PID, still wrong version
            ]
        )
        monkeypatch.setattr(probe, 'fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(probe, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(probe, 'kill_and_wait', lambda pid, timeout=5.0: None)
        monkeypatch.setattr(probe, 'spawn_detached', lambda: None)
        monkeypatch.setattr(probe, 'wait_for_health', lambda timeout=15.0: None)

        with pytest.raises(RuntimeError, match='new daemon still differs in: pxt_version'):
            probe.ensure_running()

    def test_identity_match_no_restart(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All identity fields match: ensure_running returns the base URL without killing
        or respawning anything."""
        _patch_identity(monkeypatch, {})
        monkeypatch.setattr(probe, 'fetch_health', lambda *a, **kw: _health_payload(pid=100))
        actions: list[str] = []
        monkeypatch.setattr(probe, 'kill_and_wait', lambda pid, timeout=5.0: actions.append('kill'))
        monkeypatch.setattr(probe, 'spawn_detached', lambda: actions.append('spawn'))

        url = probe.ensure_running()
        assert url.startswith('http://127.0.0.1:')
        assert actions == []

    @pytest.mark.parametrize(
        'drift_key,drift_value',
        [
            ('pxt_install_dir', '/elsewhere/site-packages/pixeltable'),
            ('python_executable', '/elsewhere/bin/python'),
            ('pixeltable_home', '/tmp/alt-home'),
            ('pixeltable_pgdata', '/tmp/alt-pgdata'),
            ('pixeltable_config_file', '/tmp/alt-config.toml'),
            ('pixeltable_env', {'PIXELTABLE_TIME_ZONE': 'America/New_York'}),
        ],
    )
    def test_each_identity_field_triggers_restart(
        self, monkeypatch: pytest.MonkeyPatch, drift_key: str, drift_value: object
    ) -> None:
        """Drift in any single identity field is sufficient to trigger a daemon restart.
        Locks in the per-field coverage so a future refactor can't silently drop one."""
        _patch_identity(monkeypatch, {})
        # Build the drifted payload by mutating after construction: spreading an
        # object-typed dict into **kwargs can't satisfy the str-typed started_at parameter
        # under mypy.
        drifted = _health_payload(pid=100)
        drifted[drift_key] = drift_value
        responses = iter([drifted, _health_payload(pid=200)])
        monkeypatch.setattr(probe, 'fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(probe, 'read_pidfile', lambda: 100)
        actions: list[tuple[str, ...] | tuple[str, int]] = []
        monkeypatch.setattr(probe, 'kill_and_wait', lambda pid, timeout=5.0: actions.append(('kill', pid)))
        monkeypatch.setattr(probe, 'spawn_detached', lambda: actions.append(('spawn',)))
        monkeypatch.setattr(probe, 'wait_for_health', lambda timeout=15.0: None)

        probe.ensure_running()
        assert actions == [('kill', 100), ('spawn',)]

    def test_pidfile_malformed(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(probe, 'pidfile_path', lambda: str(tmp_path / 'bogus.pid'))
        with open(probe.pidfile_path(), 'w', encoding='utf-8') as f:
            f.write('not-an-int')
        assert probe.read_pidfile() is None

    def test_pidfile_missing(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(probe, 'pidfile_path', lambda: str(tmp_path / 'missing.pid'))
        assert probe.read_pidfile() is None

    def test_fetch_health_rejects_non_cli_marker(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        assert probe.fetch_health() is None

    def test_fetch_health_missing_identity_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeResp:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *a: object) -> None:
                pass

            def read(self) -> bytes:
                return self._body

        # legacy daemon shape (pre-identity): missing pxt_install_dir etc. -> rejected
        legacy = json.dumps({'ok': True, 'service': 'pxt', 'pxt_version': '1.0', 'pid': 1, 'started_at': 'a'}).encode()
        monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: FakeResp(legacy))
        assert probe.fetch_health() is None
        # absent service marker / no fields at all -> also rejected
        monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: FakeResp(b'{"ok": true, "service": "pxt"}'))
        assert probe.fetch_health() is None

    def test_fetch_health_accepts_complete_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All identity fields present alongside pid/started_at -> accepted."""

        class FakeResp:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *a: object) -> None:
                pass

            def read(self) -> bytes:
                return json.dumps(_health_payload()).encode()

        monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: FakeResp())
        body = probe.fetch_health()
        assert body is not None
        assert all(k in body for k in probe._IDENTITY_KEYS)

    def test_fetch_health_url_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*a: object, **kw: object) -> None:
            raise urllib.error.URLError('refused')

        monkeypatch.setattr('urllib.request.urlopen', boom)
        assert probe.fetch_health() is None

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
        with pytest.raises(RuntimeError, match='pxt daemon log unavailable'):
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
        """If SIGTERM doesn't bring the daemon down, kill_and_wait must follow up with SIGKILL.

        Liveness is checked via os.kill(pid, 0), not /health, so a hung-but-alive daemon
        still holding the listen socket gets SIGKILLed instead of leaving the socket bound.
        """
        calls: list[int] = []

        def fake_kill(pid: int, sig: int) -> None:
            # never raises -> _pid_alive returns True every iteration, deadline expires
            calls.append(sig)

        monkeypatch.setattr(probe.os, 'kill', fake_kill)
        probe.kill_and_wait(12345, timeout=0.2)
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
        probe.kill_and_wait(12345, timeout=1.0)
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
        probe.kill_and_wait(99999)

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


class TestIdentity:
    """The identity fingerprint helpers used by ensure_running() to detect installation
    or environment drift between client and daemon."""

    def test_resolve_pixeltable_home_uses_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        assert probe._resolve_pixeltable_home() == str(tmp_path.resolve())

    def test_resolve_pixeltable_home_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('PIXELTABLE_HOME', raising=False)
        assert probe._resolve_pixeltable_home() == str(pathlib.Path('~/.pixeltable').expanduser().resolve())

    def test_resolve_pgdata_uses_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        monkeypatch.setenv('PIXELTABLE_PGDATA', str(tmp_path / 'pg'))
        assert probe._resolve_pixeltable_pgdata('/ignored') == str((tmp_path / 'pg').resolve())

    def test_resolve_pgdata_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        monkeypatch.delenv('PIXELTABLE_PGDATA', raising=False)
        assert probe._resolve_pixeltable_pgdata(str(tmp_path)) == str((tmp_path / 'pgdata').resolve())

    def test_resolve_config_file_uses_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        monkeypatch.setenv('PIXELTABLE_CONFIG', str(tmp_path / 'custom.toml'))
        assert probe._resolve_pixeltable_config_file('/ignored') == str((tmp_path / 'custom.toml').resolve())

    def test_resolve_config_file_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        monkeypatch.delenv('PIXELTABLE_CONFIG', raising=False)
        assert probe._resolve_pixeltable_config_file(str(tmp_path)) == str((tmp_path / 'config.toml').resolve())

    @pytest.mark.parametrize(
        'name,is_sensitive',
        [
            ('PIXELTABLE_HOME', False),
            ('PIXELTABLE_TIME_ZONE', False),
            ('PIXELTABLE_DB_CONNECT_STR', True),
            ('PIXELTABLE_OPENAI_API_KEY', True),
            ('PIXELTABLE_FOO_TOKEN', True),
            ('PIXELTABLE_BAR_SECRET', True),
            ('PIXELTABLE_PG_PASSWORD', True),
            ('PIXELTABLE_PG_PASSWD', True),
            # case-insensitive: lowercase still matches
            ('pixeltable_token', True),
        ],
    )
    def test_is_sensitive_env_name(self, name: str, is_sensitive: bool) -> None:
        assert probe._is_sensitive_env_name(name) is is_sensitive

    def test_redact_env_value_passthrough_for_plain(self) -> None:
        assert probe._redact_env_value('PIXELTABLE_HOME', '/x/y/z') == '/x/y/z'

    def test_redact_env_value_hashes_sensitive(self) -> None:
        v1 = probe._redact_env_value('PIXELTABLE_DB_CONNECT_STR', 'postgres://u:p@h/db')
        v2 = probe._redact_env_value('PIXELTABLE_DB_CONNECT_STR', 'postgres://u:p@h/db')
        v3 = probe._redact_env_value('PIXELTABLE_DB_CONNECT_STR', 'postgres://u:p2@h/db')
        assert v1.startswith('sha256:')
        assert 'postgres' not in v1
        # equal plaintexts -> equal hashes (the comparison invariant the client relies on)
        assert v1 == v2
        # different plaintexts -> different hashes (the drift detection invariant)
        assert v1 != v3

    def test_snapshot_filters_to_pixeltable_prefix(self) -> None:
        env = {'PIXELTABLE_HOME': '/h', 'PATH': '/usr/bin', 'OPENAI_API_KEY': 'sk-leak'}
        snap = probe._snapshot_pixeltable_env(env)
        assert snap == {'PIXELTABLE_HOME': '/h'}

    def test_snapshot_redacts_credentials(self) -> None:
        env = {'PIXELTABLE_HOME': '/h', 'PIXELTABLE_DB_CONNECT_STR': 'postgres://u:p@h/db'}
        snap = probe._snapshot_pixeltable_env(env)
        assert snap['PIXELTABLE_HOME'] == '/h'
        assert snap['PIXELTABLE_DB_CONNECT_STR'].startswith('sha256:')
        # secret value must not appear anywhere in the snapshot
        assert 'postgres' not in json.dumps(snap)

    def test_snapshot_is_deterministic(self) -> None:
        env_a = {'PIXELTABLE_B': '2', 'PIXELTABLE_A': '1'}
        env_b = {'PIXELTABLE_A': '1', 'PIXELTABLE_B': '2'}
        # Equal dicts regardless of insertion order; the comparison in ensure_running()
        # relies on this.
        assert probe._snapshot_pixeltable_env(env_a) == probe._snapshot_pixeltable_env(env_b)

    def test_identity_dict_round_trips_through_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        """A daemon serializes identity through JSON before the client sees it; a Python dict
        and the JSON-round-tripped equivalent must compare equal so equality drives the
        restart decision rather than serialization artifacts."""
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        monkeypatch.setenv('PIXELTABLE_TIME_ZONE', 'America/Los_Angeles')
        ident = probe.identity()
        assert json.loads(json.dumps(ident)) == ident

    def test_identity_diff_lists_only_changed_keys(self) -> None:
        client = dict(_DEFAULT_IDENTITY)
        daemon = {**_DEFAULT_IDENTITY, 'pixeltable_home': '/elsewhere'}
        assert probe._identity_diff(client, daemon) == ['pixeltable_home']

    def test_identity_diff_treats_missing_daemon_key_as_drift(self) -> None:
        """An old daemon that doesn't report a given identity key is treated as 'differs',
        so an outdated daemon is restarted instead of trusted."""
        client = dict(_DEFAULT_IDENTITY)
        daemon = {k: v for k, v in _DEFAULT_IDENTITY.items() if k != 'python_executable'}
        assert probe._identity_diff(client, daemon) == ['python_executable']

    def test_client_pxt_install_dir_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(name: str) -> object:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(probe.importlib.metadata, 'distribution', boom)
        assert probe._client_pxt_install_dir() is None

    def test_identity_includes_all_keys(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        """Smoke-test: identity() returns exactly the set of keys the comparison logic
        reads. A future field added to _IDENTITY_KEYS without populating it in identity()
        would silently always-mismatch; this test catches that."""
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        ident = probe.identity()
        assert set(ident.keys()) == set(probe._IDENTITY_KEYS)

    def test_identity_fails_fast_on_missing_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If importlib.metadata can't find the pixeltable distribution (broken install),
        identity() raises a clear error instead of returning a partially-None dict that
        would later cause /health to 500 and trigger a respawn loop."""
        monkeypatch.setattr(probe, '_client_pxt_version', lambda: None)
        monkeypatch.setattr(probe, '_client_pxt_install_dir', lambda: '/some/path')
        with pytest.raises(RuntimeError, match='pixeltable package metadata not found'):
            probe.identity()

        monkeypatch.setattr(probe, '_client_pxt_version', lambda: '1.0')
        monkeypatch.setattr(probe, '_client_pxt_install_dir', lambda: None)
        with pytest.raises(RuntimeError, match='pixeltable package metadata not found'):
            probe.identity()


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
        p = client_parser.Parser(prog='cli foo', epilog='Examples:\n  cli foo bar')
        p.add_argument('required')
        with pytest.raises(SystemExit) as ei:
            p.parse_args([])
        assert ei.value.code == 2
        err = capsys.readouterr().err
        assert 'cli foo' in err
        assert 'Examples:' in err

    def test_parse_cols_none(self) -> None:
        p = client_parser.Parser(prog='cli x')
        assert client_parser.parse_cols(None, p) is None

    def test_parse_cols_valid(self) -> None:
        p = client_parser.Parser(prog='cli x')
        assert client_parser.parse_cols('a,b, c', p) == ['a', 'b', 'c']

    @pytest.mark.parametrize('arg', ['a,', ',a', 'a,,b', ',', '  ,a'])
    def test_parse_cols_rejects_empty_tokens(self, arg: str, capsys: pytest.CaptureFixture) -> None:
        p = client_parser.Parser(prog='cli x')
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
        monkeypatch.setattr(client_main.sys, 'argv', ['pxt', '--help'])
        with pytest.raises(SystemExit) as ei:
            client_main.main()
        assert ei.value.code == 0
        assert 'commands:' in capsys.readouterr().out

    def test_main_no_args(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        # No command: print help and exit 0 (matches the prior `pxt` behavior so users who
        # run the script with no args get the command list, not a non-zero error code).
        monkeypatch.setattr(client_main.sys, 'argv', ['pxt'])
        with pytest.raises(SystemExit) as ei:
            client_main.main()
        assert ei.value.code == 0
        assert 'commands:' in capsys.readouterr().out

    def test_main_version_flag(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(client_main.sys, 'argv', ['pxt', '--version'])
        with pytest.raises(SystemExit) as ei:
            client_main.main()
        assert ei.value.code == 0
        out = capsys.readouterr().out
        # importlib.metadata produces the installed version; just verify the prefix is right
        # and a version-looking dotted string follows
        assert out.startswith('pxt ')
        assert '.' in out


class TestHttp:
    def test_ensure_running_failure(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        def boom() -> str:
            raise RuntimeError('cannot spawn daemon: missing fastapi')

        monkeypatch.setattr(http, 'ensure_running', boom)
        with pytest.raises(SystemExit) as ei:
            http.get('/api/health')
        assert ei.value.code == 1
        assert 'cannot spawn daemon' in capsys.readouterr().err

    def test_http_error_with_detail(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(http, 'ensure_running', lambda: 'http://127.0.0.1:1')

        def raise_http(*a: object, **kw: object) -> None:
            body = io.BytesIO(b'{"detail": "n must be > 0"}')
            raise urllib.error.HTTPError('http://x', 400, 'Bad Request', Message(), body)

        monkeypatch.setattr(http.urllib.request, 'urlopen', raise_http)
        with pytest.raises(SystemExit) as ei:
            http.post('/api/tables/t/rows', {'n': 0, 'cols': None})
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
            http.get('/api/health')
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
            http.get('/api/health')
        assert ei.value.code == 1
        assert 'cannot reach daemon' in capsys.readouterr().err


class TestShell:
    """Exercise the REPL via subprocess to cover input/eof/error branches."""

    def test_shell_runs_health_then_exits(self, pxt_daemon: int) -> None:
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon)}
        r = subprocess.run(
            ['pxt', 'shell'], input='health\nexit\n', capture_output=True, text=True, env=env, timeout=30, check=False
        )
        assert r.returncode == 0, r.stderr
        # the health response is JSON; should appear in stdout between two prompts
        assert '"service": "pxt"' in r.stdout

    def test_shell_eof_exits_cleanly(self, pxt_daemon: int) -> None:
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon)}
        r = subprocess.run(
            ['pxt', 'shell'],
            input='',  # immediate EOF
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        assert r.returncode == 0

    def test_shell_unknown_command_does_not_kill_session(self, pxt_daemon: int) -> None:
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon)}
        r = subprocess.run(
            ['pxt', 'shell'],
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
        assert '"service": "pxt"' in r.stdout

    def test_shell_rejects_nested_shell(self, pxt_daemon: int) -> None:
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon)}
        r = subprocess.run(
            ['pxt', 'shell'], input='shell\nexit\n', capture_output=True, text=True, env=env, timeout=30, check=False
        )
        assert r.returncode == 0
        assert 'already in shell' in r.stderr

    def test_shell_help(self, pxt_daemon: int) -> None:
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon)}
        r = subprocess.run(
            ['pxt', 'shell'], input='help\nexit\n', capture_output=True, text=True, env=env, timeout=30, check=False
        )
        assert r.returncode == 0
        # `help` lists every non-shell command
        assert all(c in r.stdout for c in ('health', 'ls', 'describe'))

    def test_shell_empty_line(self, pxt_daemon: int) -> None:
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon)}
        r = subprocess.run(
            ['pxt', 'shell'],
            input='\n\nhealth\nexit\n',
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        assert r.returncode == 0
        assert '"service": "pxt"' in r.stdout

    def test_shell_parse_error(self, pxt_daemon: int) -> None:
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon)}
        # unterminated quote -> shlex.split raises ValueError
        r = subprocess.run(
            ['pxt', 'shell'],
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


class TestDaemonCmd:
    """`pxt daemon start|stop|restart|status`. The action handlers in
    pxt_cli/client/commands/daemon.py thread through probe.* helpers; tests mock those at
    the boundary so they verify the command's decision logic without spawning real daemons."""

    def test_start_calls_ensure_running_and_prints(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setattr(daemon_cmd.probe, 'ensure_running', lambda: 'http://127.0.0.1:22090')
        monkeypatch.setattr(daemon_cmd.probe, 'fetch_health', lambda: {'pid': 4242})
        daemon_cmd.run(['start'])
        out = capsys.readouterr().out
        assert 'http://127.0.0.1:22090' in out
        assert '4242' in out

    def test_start_propagates_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        def boom() -> str:
            raise RuntimeError('cannot spawn daemon: missing fastapi')

        monkeypatch.setattr(daemon_cmd.probe, 'ensure_running', boom)
        with pytest.raises(SystemExit) as ei:
            daemon_cmd.run(['start'])
        assert ei.value.code == 1
        assert 'cannot spawn daemon' in capsys.readouterr().err

    def test_stop_kills_when_pid_matches(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setattr(daemon_cmd.probe, 'read_pidfile', lambda: 4242)
        monkeypatch.setattr(daemon_cmd.probe, 'fetch_health', lambda: {'pid': 4242})
        monkeypatch.setattr(daemon_cmd.probe, 'pidfile_path', lambda: str(tmp_path / 'pid'))
        killed: list[int] = []
        monkeypatch.setattr(daemon_cmd.probe, 'kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        daemon_cmd.run(['stop'])
        assert killed == [4242]
        assert 'PID 4242' in capsys.readouterr().out

    def test_stop_no_daemon_exits_1(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(daemon_cmd.probe, 'read_pidfile', lambda: None)
        monkeypatch.setattr(daemon_cmd.probe, 'fetch_health', lambda: None)
        with pytest.raises(SystemExit) as ei:
            daemon_cmd.run(['stop'])
        assert ei.value.code == 1
        assert 'no daemon running' in capsys.readouterr().err

    def test_stop_pidfile_but_no_responder_kills_tracked_pid(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: pathlib.Path
    ) -> None:
        # Daemon hung or crashed: pidfile points somewhere, /health silent. kill_and_wait
        # is idempotent on a dead PID, so the kill attempt is safe either way.
        monkeypatch.setattr(daemon_cmd.probe, 'read_pidfile', lambda: 9999)
        monkeypatch.setattr(daemon_cmd.probe, 'fetch_health', lambda: None)
        monkeypatch.setattr(daemon_cmd.probe, 'pidfile_path', lambda: str(tmp_path / 'pid'))
        killed: list[int] = []
        monkeypatch.setattr(daemon_cmd.probe, 'kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        daemon_cmd.run(['stop'])
        assert killed == [9999]
        assert 'PID 9999' in capsys.readouterr().out

    def test_stop_pid_mismatch_refuses_without_force(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setattr(daemon_cmd.probe, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(daemon_cmd.probe, 'fetch_health', lambda: {'pid': 200})
        killed: list[int] = []
        monkeypatch.setattr(daemon_cmd.probe, 'kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        with pytest.raises(SystemExit) as ei:
            daemon_cmd.run(['stop'])
        assert ei.value.code == 1
        assert 'does not match pidfile' in capsys.readouterr().err
        assert killed == []

    def test_stop_pid_mismatch_force_kills_responder(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setattr(daemon_cmd.probe, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(daemon_cmd.probe, 'fetch_health', lambda: {'pid': 200})
        monkeypatch.setattr(daemon_cmd.probe, 'pidfile_path', lambda: str(tmp_path / 'pid'))
        killed: list[int] = []
        monkeypatch.setattr(daemon_cmd.probe, 'kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        daemon_cmd.run(['stop', '--force'])
        # --force on mismatch kills the responder, not the tracked pidfile PID
        assert killed == [200]

    def test_stop_responder_without_pidfile_refuses_without_force(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setattr(daemon_cmd.probe, 'read_pidfile', lambda: None)
        monkeypatch.setattr(daemon_cmd.probe, 'fetch_health', lambda: {'pid': 200})
        killed: list[int] = []
        monkeypatch.setattr(daemon_cmd.probe, 'kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        with pytest.raises(SystemExit) as ei:
            daemon_cmd.run(['stop'])
        assert ei.value.code == 1
        assert 'no pidfile' in capsys.readouterr().err
        assert killed == []

    def test_status_prints_identity_text(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(
            daemon_cmd.probe,
            'fetch_health',
            lambda: {
                'pid': 4242,
                'started_at': '2026-05-18T12:00:00+00:00',
                'service': 'pxt',
                'pxt_version': '1.2.3',
                'pxt_install_dir': '/p/dir',
                'python_executable': '/p/bin/python',
                'pixeltable_home': '/p/home',
                'pixeltable_pgdata': '/p/home/pgdata',
                'pixeltable_config_file': '/p/home/config.toml',
                'pixeltable_env': {'PIXELTABLE_TIME_ZONE': 'America/Los_Angeles'},
            },
        )
        daemon_cmd.run(['status'])
        out = capsys.readouterr().out
        assert 'PID' in out
        assert '4242' in out
        assert '1.2.3' in out
        assert 'PIXELTABLE_TIME_ZONE' in out

    def test_status_json_is_raw_dict(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        payload = {'pid': 4242, 'service': 'pxt', 'pixeltable_env': {}}
        monkeypatch.setattr(daemon_cmd.probe, 'fetch_health', lambda: payload)
        daemon_cmd.run(['status', '--json'])
        assert json.loads(capsys.readouterr().out) == payload

    def test_status_no_daemon_exits_1(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(daemon_cmd.probe, 'fetch_health', lambda: None)
        with pytest.raises(SystemExit) as ei:
            daemon_cmd.run(['status'])
        assert ei.value.code == 1
        assert 'no daemon running' in capsys.readouterr().err

    def test_restart_stops_then_starts(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: pathlib.Path
    ) -> None:
        # First cycle: daemon present with matching pids; stop kills it, then start spawns
        # a new one. fetch_health returns a daemon for stop, then a daemon (different PID)
        # after start.
        states = iter(
            [
                {'pid': 100},  # for the stop branch
                {'pid': 200},  # for the start branch's post-spawn lookup
            ]
        )
        monkeypatch.setattr(daemon_cmd.probe, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(daemon_cmd.probe, 'fetch_health', lambda: next(states))
        monkeypatch.setattr(daemon_cmd.probe, 'pidfile_path', lambda: str(tmp_path / 'pid'))
        actions: list[str] = []
        monkeypatch.setattr(daemon_cmd.probe, 'kill_and_wait', lambda pid, timeout=5.0: actions.append(f'kill:{pid}'))

        def fake_ensure_running() -> str:
            actions.append('spawn')
            return 'http://127.0.0.1:22090'

        monkeypatch.setattr(daemon_cmd.probe, 'ensure_running', fake_ensure_running)

        daemon_cmd.run(['restart'])
        assert actions == ['kill:100', 'spawn']
        out = capsys.readouterr().out
        assert 'http://127.0.0.1:22090' in out
        assert '200' in out

    def test_restart_with_no_existing_daemon(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: pathlib.Path
    ) -> None:
        # Nothing to stop initially; restart should still proceed to start without erroring.
        states = iter([None, {'pid': 200}])
        monkeypatch.setattr(daemon_cmd.probe, 'read_pidfile', lambda: None)
        monkeypatch.setattr(daemon_cmd.probe, 'fetch_health', lambda: next(states))
        monkeypatch.setattr(daemon_cmd.probe, 'ensure_running', lambda: 'http://127.0.0.1:22090')
        monkeypatch.setattr(daemon_cmd.probe, 'pidfile_path', lambda: str(tmp_path / 'pid'))
        daemon_cmd.run(['restart'])
        out = capsys.readouterr().out
        assert 'http://127.0.0.1:22090' in out
