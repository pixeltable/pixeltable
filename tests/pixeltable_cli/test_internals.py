"""Unit tests for cli internals.

Covers things that aren't reachable through the daemon smoke tests:
  - client_utils.py spawn / restart / kill safety paths (monkeypatched)
  - the confirm.py interactive prompt
  - parser.py / main.py error and help paths
  - the client HTTP get/post error branches
  - the interactive shell REPL (driven via subprocess.Popen)
  - management_client's retry of a dropped management API connection
"""

import http.client
import importlib.metadata
import io
import json
import os
import pathlib
import platform
import signal
import socket
import subprocess
import sys
import urllib.error
from collections.abc import Callable, Iterator
from email.message import Message
from types import ModuleType, SimpleNamespace
from typing import Any, Self
from unittest.mock import patch

import pydantic
import pytest
import requests

from pixeltable import exceptions as excs, metadata
from pixeltable.catalog import Path as PxtPath
from pixeltable.config import Config
from pixeltable.service import db, management_client
from pixeltable.service.management_protocol import (
    CreateDbRequest,
    DeleteDbRequest,
    GetDbRequest,
    ListDbRequest,
    ListOrgsRequest,
    ManagementOperationType,
    StartDbRequest,
    StopDbRequest,
)
from pixeltable.serving.service_instance import ServiceInstanceState
from pixeltable.utils.app_module import load_app_module, module_routers, service_spec, services_by_name
from pixeltable.utils.project import project_fingerprint
from pixeltable_cli import utils
from pixeltable_cli.client import hosted, main as client_main, parser as client_parser, utils as client_utils
from pixeltable_cli.client.commands import (
    daemon as daemon_cmd,
    db as db_cmd,
    org as org_cmd,
    service as service_cmd,
    shell as shell_cmd,
    status as status_cmd,
)
from pixeltable_cli.server import daemon as server_daemon, router as server_router, routes as server_routes
from tests.utils import pxt_raises, skip_test_if_not_installed


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


# A canonical identity dict used by the identity-equality tests below. Real values are too tied to the
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
    """Pin utils.identity() to a known dict so tests don't depend on the host environment."""
    ident = {**_DEFAULT_IDENTITY, **overrides}
    monkeypatch.setattr(client_utils, 'identity', lambda: dict(ident))
    # pin project_root to None to avoid daemon restarts
    monkeypatch.setattr(client_utils, 'project_root', lambda: None)
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
        pid = client_utils.read_pidfile()
        if pid is not None:
            # Reuse the production kill helper: it already handles the SIGKILL fallback
            # and the Windows quirks around os.kill(pid, 0). Cleanup is best-effort.
            try:
                client_utils.kill_and_wait(pid, timeout=3.0)
            except Exception:
                pass
        if prior is None:
            os.environ.pop('PXT_PORT', None)
        else:
            os.environ['PXT_PORT'] = prior


class TestProjectRootParity:
    """The client duplicates the library's project-root rule; the two copies have to agree."""

    def test_copies_agree(self, tmp_path: pathlib.Path) -> None:
        from pixeltable.config import _find_project_root as library_copy
        from pixeltable_cli.utils import find_project_root as client_copy

        nested = tmp_path / 'proj' / 'ad_gen'
        nested.mkdir(parents=True)
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()

        def both(start: pathlib.Path) -> tuple[pathlib.Path | None, pathlib.Path | None]:
            return library_copy(start), client_copy(start)

        # no config anywhere above
        library, client = both(nested)
        assert library == client is None

        # a pyproject.toml without the section, then with it
        (tmp_path / 'proj' / 'pyproject.toml').write_text('[project]\nname = "proj"\n')
        library, client = both(nested)
        assert library == client is None
        (tmp_path / 'proj' / 'pyproject.toml').write_text('[project]\nname = "proj"\n\n[tool.pixeltable]\n')
        library, client = both(nested)
        assert library == client == tmp_path / 'proj'

        # the nearest config wins, and a pixeltable.toml beats a pyproject.toml beside it
        (nested / 'pyproject.toml').write_text('[tool.pixeltable]\n')
        (nested / 'pixeltable.toml').write_text('')
        library, client = both(nested)
        assert library == client == nested

        # a file that cannot be parsed might be the project config, so both copies refuse rather than
        # resolving against a directory above it
        (elsewhere / 'pyproject.toml').write_text('[tool.pixeltable\n')
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r'cannot be parsed'):
            library_copy(elsewhere)
        with pytest.raises(RuntimeError, match=r'cannot be parsed'):
            client_copy(elsewhere)

        # the same, with a project root above the unreadable file
        unreadable = tmp_path / 'proj' / 'unreadable'
        unreadable.mkdir()
        (unreadable / 'pyproject.toml').write_text('[tool.pixeltable\n')
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r'cannot be parsed'):
            library_copy(unreadable)
        with pytest.raises(RuntimeError, match=r'cannot be parsed'):
            client_copy(unreadable)


class TestProbe:
    """Spawn / restart / kill safety paths."""

    def test_auto_spawn_no_daemon(self, fresh_port: int) -> None:
        """Cold start: no daemon on the port, the cli client spawns one and routes the command."""
        env = {**os.environ, 'PXT_PORT': str(fresh_port)}
        r = subprocess.run(
            [sys.executable, '-m', 'pixeltable_cli.client.main', 'health'],
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
        assert client_utils.read_pidfile() == body['pid']

    def test_no_pidfile_spawns(self) -> None:
        """Cold start with no pidfile: spawn straight away, nothing to reclaim."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(client_utils, 'fetch_health', lambda *a, **kw: None)
            m.setattr(client_utils, 'read_pidfile', lambda: None)
            actions: list[str] = []
            m.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: actions.append('kill'))
            m.setattr(client_utils, 'spawn_detached', lambda: actions.append('spawn'))
            m.setattr(client_utils, 'wait_for_health', lambda timeout=15.0: None)

            client_utils.ensure_running()
            assert actions == ['spawn']

    def test_hung_daemon_reclaimed(self) -> None:
        """A daemon we started is alive (pidfile names a live PID that is one of ours) but stays
        silent past the grace window: it is hung, so kill it and spawn a replacement instead of
        failing to bind."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(client_utils, 'fetch_health', lambda *a, **kw: None)
            m.setattr(client_utils, 'read_pidfile', lambda: 100)
            m.setattr(client_utils, '_pid_alive', lambda pid: True)
            m.setattr(client_utils, '_pid_is_our_daemon', lambda pid: True)
            m.setattr(client_utils, '_await_health', lambda timeout: False)
            actions: list[tuple[str, int] | str] = []
            m.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: actions.append(('kill', pid)))
            m.setattr(client_utils, 'spawn_detached', lambda: actions.append('spawn'))
            m.setattr(client_utils, 'wait_for_health', lambda timeout=15.0: None)

            client_utils.ensure_running()
            assert actions == [('kill', 100), 'spawn']

    def test_foreign_live_pid_kept(self) -> None:
        """The pidfile names a live PID, but the process is not one of our daemons (the PID was
        recycled after our daemon exited). It must be treated as a stale pidfile: do not kill the
        unrelated process, just spawn a fresh daemon."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(client_utils, 'fetch_health', lambda *a, **kw: None)
            m.setattr(client_utils, 'read_pidfile', lambda: 100)
            m.setattr(client_utils, '_pid_alive', lambda pid: True)
            m.setattr(client_utils, '_pid_is_our_daemon', lambda pid: False)
            actions: list[str] = []
            m.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: actions.append('kill'))
            m.setattr(client_utils, '_await_health', lambda timeout: pytest.fail('grace window must be skipped'))
            m.setattr(client_utils, 'spawn_detached', lambda: actions.append('spawn'))
            m.setattr(client_utils, 'wait_for_health', lambda timeout=15.0: None)

            client_utils.ensure_running()
            assert actions == ['spawn']

    def test_slow_daemon_kept(self) -> None:
        """A daemon we started is alive but still importing pixeltable; it answers health within the
        grace window. It must be used as-is, not killed as if it were hung."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(client_utils, 'fetch_health', lambda *a, **kw: None)
            m.setattr(client_utils, 'read_pidfile', lambda: 100)
            m.setattr(client_utils, '_pid_alive', lambda pid: True)
            m.setattr(client_utils, '_pid_is_our_daemon', lambda pid: True)
            m.setattr(client_utils, '_await_health', lambda timeout: True)
            actions: list[str] = []
            m.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: actions.append('kill'))
            m.setattr(client_utils, 'spawn_detached', lambda: actions.append('spawn'))

            url = client_utils.ensure_running()
            assert url == client_utils.base_url()
            assert actions == []

    def test_dead_pidfile_spawns(self) -> None:
        """A stale pidfile naming a PID that is no longer alive (port already released): no reclaim,
        just spawn."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(client_utils, 'fetch_health', lambda *a, **kw: None)
            m.setattr(client_utils, 'read_pidfile', lambda: 100)
            m.setattr(client_utils, '_pid_alive', lambda pid: False)
            actions: list[str] = []
            m.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: actions.append('kill'))
            m.setattr(client_utils, 'spawn_detached', lambda: actions.append('spawn'))
            m.setattr(client_utils, 'wait_for_health', lambda timeout=15.0: None)

            client_utils.ensure_running()
            assert actions == ['spawn']

    def test_identity_mismatch_no_pidfile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Version drift restarts the daemon even when the pidfile is missing/corrupt: the health response
        identifies the responder as ours, so its self-reported PID is the one terminated (no pidfile needed)."""
        _patch_identity(monkeypatch, {'pxt_version': 'NEW'})
        responses = iter([_health_payload(pxt_version='OLD', pid=99999), _health_payload(pxt_version='NEW', pid=200)])
        monkeypatch.setattr(client_utils, 'fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(client_utils, 'read_pidfile', lambda: None)  # pidfile lost/corrupt
        actions: list[tuple[str, int] | tuple[str, ...]] = []
        monkeypatch.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: actions.append(('kill', pid)))
        monkeypatch.setattr(client_utils, 'spawn_detached', lambda: actions.append(('spawn',)))
        monkeypatch.setattr(client_utils, 'wait_for_health', lambda timeout=15.0: None)

        client_utils.ensure_running()
        assert actions == [('kill', 99999), ('spawn',)]

    def test_identity_mismatch_restart_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Matching pidfile + identity drift: ensure_running kills the old daemon, spawns a
        new one, and cross-verifies the post-restart responder's identity matches ours."""
        _patch_identity(monkeypatch, {'pxt_version': 'NEW'})
        responses = iter([_health_payload(pxt_version='OLD', pid=100), _health_payload(pxt_version='NEW', pid=200)])
        monkeypatch.setattr(client_utils, 'fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(client_utils, 'read_pidfile', lambda: 100)
        actions: list[tuple[str, ...] | tuple[str, int]] = []
        monkeypatch.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: actions.append(('kill', pid)))
        monkeypatch.setattr(client_utils, 'spawn_detached', lambda: actions.append(('spawn',)))
        monkeypatch.setattr(client_utils, 'wait_for_health', lambda timeout=15.0: None)

        url = client_utils.ensure_running()
        assert url.startswith('http://127.0.0.1:')
        assert actions == [('kill', 100), ('spawn',)]

    def test_identity_mismatch_invalid_pid_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Identity drift but the responder reports a non-int pid: refuse to restart (no kill, no spawn)
        rather than act on an untrustworthy pid."""
        _patch_identity(monkeypatch, {'pxt_version': 'NEW'})
        health = _health_payload(pxt_version='OLD')
        health['pid'] = 'not-a-pid'
        monkeypatch.setattr(client_utils, 'fetch_health', lambda *a, **kw: health)
        monkeypatch.setattr(client_utils, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(
            client_utils, 'kill_and_wait', lambda pid, timeout=5.0: pytest.fail('must not kill an invalid pid')
        )
        monkeypatch.setattr(client_utils, 'spawn_detached', lambda: pytest.fail('must not spawn'))

        with pytest.raises(RuntimeError, match='invalid pid'):
            client_utils.ensure_running()

    def test_cross_verify_kept_killed_pid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Post-restart cross-verify: the new responder still reports the killed PID."""
        _patch_identity(monkeypatch, {'pxt_version': 'NEW'})
        responses = iter([_health_payload(pxt_version='OLD', pid=100), _health_payload(pxt_version='NEW', pid=100)])
        monkeypatch.setattr(client_utils, 'fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(client_utils, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: None)
        monkeypatch.setattr(client_utils, 'spawn_detached', lambda: None)
        monkeypatch.setattr(client_utils, 'wait_for_health', lambda timeout=15.0: None)

        with pytest.raises(RuntimeError, match='new daemon kept the killed PID 100'):
            client_utils.ensure_running()

    def test_cross_verify_no_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Post-restart cross-verify: the new daemon never responds to /health."""
        _patch_identity(monkeypatch, {'pxt_version': 'NEW'})
        responses = iter([_health_payload(pxt_version='OLD', pid=100), None])
        monkeypatch.setattr(client_utils, 'fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(client_utils, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: None)
        monkeypatch.setattr(client_utils, 'spawn_detached', lambda: None)
        monkeypatch.setattr(client_utils, 'wait_for_health', lambda timeout=15.0: None)

        with pytest.raises(RuntimeError, match='new daemon did not respond'):
            client_utils.ensure_running()

    def test_cross_verify_identity_still_differs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Post-restart cross-verify: a fresh PID came up but still has the wrong identity."""
        _patch_identity(monkeypatch, {'pxt_version': 'NEW'})
        responses = iter(
            [
                _health_payload(pxt_version='OLD', pid=100),
                _health_payload(pxt_version='OLD', pid=200),  # fresh PID, still wrong version
            ]
        )
        monkeypatch.setattr(client_utils, 'fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(client_utils, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: None)
        monkeypatch.setattr(client_utils, 'spawn_detached', lambda: None)
        monkeypatch.setattr(client_utils, 'wait_for_health', lambda timeout=15.0: None)

        with pytest.raises(RuntimeError, match='new daemon still differs in: pxt_version'):
            client_utils.ensure_running()

    def test_identity_match_no_restart(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All identity fields match: ensure_running returns the base URL without killing
        or respawning anything."""
        _patch_identity(monkeypatch, {})
        monkeypatch.setattr(client_utils, 'fetch_health', lambda *a, **kw: _health_payload(pid=100))
        actions: list[str] = []
        monkeypatch.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: actions.append('kill'))
        monkeypatch.setattr(client_utils, 'spawn_detached', lambda: actions.append('spawn'))

        url = client_utils.ensure_running()
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
    def test_identity_field_restarts(
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
        monkeypatch.setattr(client_utils, 'fetch_health', lambda *a, **kw: next(responses))
        monkeypatch.setattr(client_utils, 'read_pidfile', lambda: 100)
        actions: list[tuple[str, ...] | tuple[str, int]] = []
        monkeypatch.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: actions.append(('kill', pid)))
        monkeypatch.setattr(client_utils, 'spawn_detached', lambda: actions.append(('spawn',)))
        monkeypatch.setattr(client_utils, 'wait_for_health', lambda timeout=15.0: None)

        client_utils.ensure_running()
        assert actions == [('kill', 100), ('spawn',)]

    def test_pidfile_malformed(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(client_utils, 'pidfile_path', lambda: str(tmp_path / 'bogus.pid'))
        with open(client_utils.pidfile_path(), 'w', encoding='utf-8') as f:
            f.write('not-an-int')
        assert client_utils.read_pidfile() is None

    def test_pidfile_missing(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(client_utils, 'pidfile_path', lambda: str(tmp_path / 'missing.pid'))
        assert client_utils.read_pidfile() is None

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
        assert client_utils.fetch_health() is None

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
        assert client_utils.fetch_health() is None
        # absent service marker / no fields at all -> also rejected
        monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: FakeResp(b'{"ok": true, "service": "pxt"}'))
        assert client_utils.fetch_health() is None

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
        body = client_utils.fetch_health()
        assert body is not None
        assert all(k in body for k in utils._IDENTITY_KEYS)

    def test_fetch_health_url_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*a: object, **kw: object) -> None:
            raise urllib.error.URLError('refused')

        monkeypatch.setattr('urllib.request.urlopen', boom)
        assert client_utils.fetch_health() is None

    def test_fetch_health_truncated_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A daemon that sends headers and then drops the connection isn't healthy, it isn't a crash."""

        class FakeResp:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *a: object) -> None:
                pass

            def read(self) -> bytes:
                raise http.client.IncompleteRead(b'', 1020)

        monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: FakeResp())
        assert client_utils.fetch_health() is None

    def test_client_pxt_version_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(utils.importlib.metadata, 'version', boom)
        assert utils._pxt_version() is None

    def test_spawn_detached_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*a: object, **kw: object) -> None:
            raise OSError('disk full')

        monkeypatch.setattr(client_utils.os, 'makedirs', boom)
        with pytest.raises(RuntimeError, match='pxt daemon log unavailable'):
            client_utils.spawn_detached()

    def test_spawn_detached_cwd_off_sys_path(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # `python -m` puts the daemon's working directory at the front of sys.path. Pinning cwd to
        # the pixeltable home and setting PYTHONSAFEPATH keeps a pixeltable/ folder in the directory
        # pxt was invoked from out of the daemon's import path.
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        calls: list[tuple[list[str], dict[str, Any]]] = []

        def fake_popen(args: list[str], **kwargs: Any) -> None:
            calls.append((args, kwargs))

        monkeypatch.setattr(client_utils.subprocess, 'Popen', fake_popen)
        client_utils.spawn_detached()

        args, kwargs = calls[0]
        assert args[:3] == [sys.executable, '-m', 'pixeltable_cli.server.daemon']
        assert kwargs['cwd'] == client_utils._resolve_pixeltable_home()
        assert kwargs['env']['PYTHONSAFEPATH'] == '1'

    def test_tail_daemon_log_missing(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        # log file does not exist -> empty string, no exception
        assert client_utils._tail_daemon_log() == ''

    def test_tail_daemon_log_truncates(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        log_path = client_utils._daemon_log_path()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            for i in range(50):
                f.write(f'line {i}\n')
        tail = client_utils._tail_daemon_log(n_lines=3)
        assert tail.splitlines() == ['line 47', 'line 48', 'line 49']

    def test_wait_for_health_timeout_log_tail(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        log_path = client_utils._daemon_log_path()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('startup blew up: address already in use\n')
        monkeypatch.setattr(client_utils, 'is_running', lambda timeout=0.3: False)
        with pytest.raises(RuntimeError, match='did not come up') as ei:
            client_utils.wait_for_health(timeout=0.2)
        assert 'address already in use' in str(ei.value)

    def test_kill_wait_sigkill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If SIGTERM doesn't bring the daemon down, kill_and_wait must follow up with SIGKILL.

        Liveness is checked via os.kill(pid, 0), not /health, so a hung-but-alive daemon
        still holding the listen socket gets SIGKILLed instead of leaving the socket bound.
        """
        calls: list[int] = []

        def fake_kill(pid: int, sig: int) -> None:
            # never raises -> _pid_alive returns True every iteration, deadline expires
            calls.append(sig)

        monkeypatch.setattr(client_utils.os, 'kill', fake_kill)
        client_utils.kill_and_wait(12345, timeout=0.2)
        assert signal.SIGTERM in calls
        # On non-Windows we have a real SIGKILL; on Windows it falls back to SIGTERM.
        assert getattr(signal, 'SIGKILL', signal.SIGTERM) in calls

    def test_kill_wait_pid_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIGTERM goes through, then the PID exits -> return without escalating to SIGKILL."""
        calls: list[int] = []

        def fake_kill(pid: int, sig: int) -> None:
            calls.append(sig)
            # signal 0 is the liveness probe: raise to simulate the PID being gone
            if sig == 0:
                raise ProcessLookupError

        monkeypatch.setattr(client_utils.os, 'kill', fake_kill)
        client_utils.kill_and_wait(12345, timeout=1.0)
        sigkill = getattr(signal, 'SIGKILL', signal.SIGTERM)
        assert signal.SIGTERM in calls
        # On platforms where SIGKILL != SIGTERM, it must NOT have been issued.
        if sigkill != signal.SIGTERM:
            assert sigkill not in calls

    def test_kill_wait_no_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(pid: int, sig: int) -> None:
            raise ProcessLookupError

        monkeypatch.setattr(client_utils.os, 'kill', boom)
        # Should return cleanly without raising
        client_utils.kill_and_wait(99999)

    def test_pid_alive_dead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(pid: int, sig: int) -> None:
            raise ProcessLookupError

        monkeypatch.setattr(client_utils.os, 'kill', boom)
        assert client_utils._pid_alive(99999) is False

    def test_pid_alive_alive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(client_utils.os, 'kill', lambda pid, sig: None)
        assert client_utils._pid_alive(12345) is True

    def test_pid_alive_permission_denied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PermissionError means the PID exists but is owned by another user; treat as alive."""

        def boom(pid: int, sig: int) -> None:
            raise PermissionError

        monkeypatch.setattr(client_utils.os, 'kill', boom)
        assert client_utils._pid_alive(1) is True

    def test_pid_alive_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(pid: int, sig: int) -> None:
            raise OSError('einval')

        monkeypatch.setattr(client_utils.os, 'kill', boom)
        assert client_utils._pid_alive(0) is False

    def test_pid_is_our_daemon_matches_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            client_utils, '_pid_cmdline', lambda pid: '/opt/conda/bin/python -m pixeltable_cli.server.daemon'
        )
        assert client_utils._pid_is_our_daemon(100) is True

    def test_pid_is_our_daemon_rejects_unrelated_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A recycled PID running something else must not be mistaken for our daemon."""
        monkeypatch.setattr(client_utils, '_pid_cmdline', lambda pid: '/usr/bin/vim notes.txt')
        assert client_utils._pid_is_our_daemon(100) is False

    def test_pid_is_our_daemon_rejects_substring(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A recycled PID whose argv merely mentions the module name (without the `-m <module>`
        launch form) must not be mistaken for our daemon."""
        monkeypatch.setattr(
            client_utils, '_pid_cmdline', lambda pid: 'python -c import pixeltable_cli.server.daemon as d'
        )
        assert client_utils._pid_is_our_daemon(100) is False

    def test_pid_is_our_daemon_no_cmdline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the command line can't be read, ownership is unconfirmed -> treat as not ours."""
        monkeypatch.setattr(client_utils, '_pid_cmdline', lambda pid: None)
        assert client_utils._pid_is_our_daemon(100) is False

    @pytest.mark.skipif(platform.system() == 'Windows', reason='_pid_cmdline has no stdlib argv source on Windows')
    def test_pid_cmdline_reads_self(self) -> None:
        """On POSIX the running interpreter's own command line is readable and mentions python."""
        cmdline = client_utils._pid_cmdline(os.getpid())
        assert cmdline is not None
        assert 'python' in cmdline.lower() or 'pytest' in cmdline.lower()

    def test_pid_cmdline_none_on_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows has no cheap stdlib argv source, so _pid_cmdline reports unknown (None) and
        the caller falls back to refusing to kill rather than reclaiming."""
        monkeypatch.setattr(client_utils, '_IS_WINDOWS', True)

        def no_proc(_p: str, *a: object, **kw: object) -> None:
            raise OSError('no /proc on windows')

        monkeypatch.setattr('builtins.open', no_proc)
        assert client_utils._pid_cmdline(os.getpid()) is None


class TestIdentity:
    """The identity fingerprint helpers used by ensure_running() to detect installation
    or environment drift between client and daemon."""

    def test_resolve_pixeltable_home_uses_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        assert utils._resolve_pixeltable_home() == str(tmp_path.resolve())

    def test_resolve_pixeltable_home_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('PIXELTABLE_HOME', raising=False)
        assert utils._resolve_pixeltable_home() == str(pathlib.Path('~/.pixeltable').expanduser().resolve())

    def test_resolve_pgdata_uses_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        monkeypatch.setenv('PIXELTABLE_PGDATA', str(tmp_path / 'pg'))
        assert utils._resolve_pixeltable_pgdata('/ignored') == str((tmp_path / 'pg').resolve())

    def test_resolve_pgdata_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        monkeypatch.delenv('PIXELTABLE_PGDATA', raising=False)
        assert utils._resolve_pixeltable_pgdata(str(tmp_path)) == str((tmp_path / 'pgdata').resolve())

    def test_resolve_config_file_uses_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        monkeypatch.setenv('PIXELTABLE_CONFIG', str(tmp_path / 'custom.toml'))
        assert utils._resolve_pixeltable_config_file('/ignored') == str((tmp_path / 'custom.toml').resolve())

    def test_resolve_config_file_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        monkeypatch.delenv('PIXELTABLE_CONFIG', raising=False)
        assert utils._resolve_pixeltable_config_file(str(tmp_path)) == str((tmp_path / 'config.toml').resolve())

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
        assert utils._is_sensitive_env_name(name) is is_sensitive

    def test_redact_env_value_plain(self) -> None:
        assert utils._redact_env_value('PIXELTABLE_HOME', '/x/y/z') == '/x/y/z'

    def test_redact_env_value_hashes_sensitive(self) -> None:
        v1 = utils._redact_env_value('PIXELTABLE_DB_CONNECT_STR', 'postgres://u:p@h/db')
        v2 = utils._redact_env_value('PIXELTABLE_DB_CONNECT_STR', 'postgres://u:p@h/db')
        v3 = utils._redact_env_value('PIXELTABLE_DB_CONNECT_STR', 'postgres://u:p2@h/db')
        assert v1.startswith('sha256:')
        assert 'postgres' not in v1
        # equal plaintexts -> equal hashes (the comparison invariant the client relies on)
        assert v1 == v2
        # different plaintexts -> different hashes (the drift detection invariant)
        assert v1 != v3

    def test_snapshot_pixeltable_prefix(self) -> None:
        env = {'PIXELTABLE_HOME': '/h', 'PATH': '/usr/bin', 'OPENAI_API_KEY': 'sk-leak'}
        snap = utils._snapshot_pixeltable_env(env)
        assert snap == {'PIXELTABLE_HOME': '/h'}

    def test_snapshot_redacts_credentials(self) -> None:
        env = {'PIXELTABLE_HOME': '/h', 'PIXELTABLE_DB_CONNECT_STR': 'postgres://u:p@h/db'}
        snap = utils._snapshot_pixeltable_env(env)
        assert snap['PIXELTABLE_HOME'] == '/h'
        assert snap['PIXELTABLE_DB_CONNECT_STR'].startswith('sha256:')
        # secret value must not appear anywhere in the snapshot
        assert 'postgres' not in json.dumps(snap)

    def test_snapshot_is_deterministic(self) -> None:
        env_a = {'PIXELTABLE_B': '2', 'PIXELTABLE_A': '1'}
        env_b = {'PIXELTABLE_A': '1', 'PIXELTABLE_B': '2'}
        # Equal dicts regardless of insertion order; the comparison in ensure_running()
        # relies on this.
        assert utils._snapshot_pixeltable_env(env_a) == utils._snapshot_pixeltable_env(env_b)

    def test_identity_dict_json_round_trip(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        """A daemon serializes identity through JSON before the client sees it; a Python dict
        and the JSON-round-tripped equivalent must compare equal so equality drives the
        restart decision rather than serialization artifacts."""
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        monkeypatch.setenv('PIXELTABLE_TIME_ZONE', 'America/Los_Angeles')
        ident = utils.identity()
        assert json.loads(json.dumps(ident)) == ident

    def test_identity_diff_changed_keys(self) -> None:
        client = dict(_DEFAULT_IDENTITY)
        daemon = {**_DEFAULT_IDENTITY, 'pixeltable_home': '/elsewhere'}
        assert client_utils._identity_diff(client, daemon) == ['pixeltable_home']

    def test_identity_diff_missing_daemon_key(self) -> None:
        """An old daemon that doesn't report a given identity key is treated as 'differs',
        so an outdated daemon is restarted instead of trusted."""
        client = dict(_DEFAULT_IDENTITY)
        daemon = {k: v for k, v in _DEFAULT_IDENTITY.items() if k != 'python_executable'}
        assert client_utils._identity_diff(client, daemon) == ['python_executable']

    def test_client_pxt_install_dir_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(name: str) -> object:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(utils.importlib.metadata, 'distribution', boom)
        assert utils._pxt_install_dir() is None

    def test_identity_all_keys(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        """Smoke-test: identity() returns exactly the set of keys the comparison logic
        reads. A future field added to _IDENTITY_KEYS without populating it in identity()
        would silently always-mismatch; this test catches that."""
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        ident = utils.identity()
        assert set(ident.keys()) == set(utils._IDENTITY_KEYS)

    def test_identity_missing_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If importlib.metadata can't find the pixeltable distribution (broken install),
        identity() raises a clear error instead of returning a partially-None dict that
        would later cause /health to 500 and trigger a respawn loop."""
        monkeypatch.setattr(utils, '_pxt_version', lambda: None)
        monkeypatch.setattr(utils, '_pxt_install_dir', lambda: '/some/path')
        with pytest.raises(RuntimeError, match='pixeltable package metadata not found'):
            utils.identity()

        monkeypatch.setattr(utils, '_pxt_version', lambda: '1.0')
        monkeypatch.setattr(utils, '_pxt_install_dir', lambda: None)
        with pytest.raises(RuntimeError, match='pixeltable package metadata not found'):
            utils.identity()


class TestConfirm:
    def test_force_short_circuits(self) -> None:
        # No TTY, no input - force=True must just return.
        client_utils.confirm_or_exit('drop something?', force=True)

    def test_no_tty_refuses(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(client_utils, 'stdin_is_a_tty', lambda: False)
        with pytest.raises(SystemExit) as ei:
            client_utils.confirm_or_exit('drop something?', force=False)
        assert ei.value.code == 2
        assert '--force' in capsys.readouterr().err

    def test_tty_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(client_utils, 'stdin_is_a_tty', lambda: True)
        monkeypatch.setattr(client_utils.sys, 'stdin', io.StringIO('y\n'))
        # Should not raise.
        client_utils.confirm_or_exit('drop something?', force=False)

    def test_tty_no(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(client_utils, 'stdin_is_a_tty', lambda: True)
        monkeypatch.setattr(client_utils.sys, 'stdin', io.StringIO('n\n'))
        with pytest.raises(SystemExit) as ei:
            client_utils.confirm_or_exit('drop something?', force=False, refused_exit_code=3)
        # answering no is refusal, same as the non-tty path
        assert ei.value.code == 3
        assert 'aborted' in capsys.readouterr().err

    def test_tty_empty_aborts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(client_utils, 'stdin_is_a_tty', lambda: True)
        monkeypatch.setattr(client_utils.sys, 'stdin', io.StringIO('\n'))
        with pytest.raises(SystemExit):
            client_utils.confirm_or_exit('drop something?', force=False)

    def test_stdin_tty_posix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-Windows path: isatty() True -> returns True without touching ctypes."""

        class FakeStdin:
            def isatty(self) -> bool:
                return True

        monkeypatch.setattr(client_utils.sys, 'stdin', FakeStdin())
        monkeypatch.setattr(client_utils.sys, 'platform', 'linux')
        assert client_utils.stdin_is_a_tty() is True

    def test_stdin_tty_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeStdin:
            def isatty(self) -> bool:
                return False

        monkeypatch.setattr(client_utils.sys, 'stdin', FakeStdin())
        assert client_utils.stdin_is_a_tty() is False


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
    def test_help_lists_commands(self, capsys: pytest.CaptureFixture) -> None:
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
        # No command: print help and exit 0 so users who run the script with no args get
        # the command list, not a non-zero error code.
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
            raise RuntimeError('cannot spawn daemon: simulated failure')

        monkeypatch.setattr(client_utils, 'ensure_running', boom)
        with pytest.raises(SystemExit) as ei:
            client_utils.get_request('/api/health')
        assert ei.value.code == 1
        assert 'cannot spawn daemon' in capsys.readouterr().err

    def test_http_error_with_detail(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(client_utils, 'ensure_running', lambda: 'http://127.0.0.1:1')

        def raise_http(*a: object, **kw: object) -> None:
            body = io.BytesIO(b'{"detail": "n must be > 0"}')
            raise urllib.error.HTTPError('http://x', 400, 'Bad Request', Message(), body)

        monkeypatch.setattr(client_utils.urllib.request, 'urlopen', raise_http)
        with pytest.raises(SystemExit) as ei:
            client_utils.post_request('/api/tables/t/rows', {'n': 0, 'cols': None})
        assert ei.value.code == 1
        err = capsys.readouterr().err
        assert '400' in err
        assert 'n must be > 0' in err

    def test_http_error_unparseable_body(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(client_utils, 'ensure_running', lambda: 'http://127.0.0.1:1')

        def raise_http(*a: object, **kw: object) -> None:
            raise urllib.error.HTTPError(
                'http://x', 500, 'Internal Server Error', Message(), io.BytesIO(b'<html>not json</html>')
            )

        monkeypatch.setattr(client_utils.urllib.request, 'urlopen', raise_http)
        with pytest.raises(SystemExit) as ei:
            client_utils.get_request('/api/health')
        assert ei.value.code == 1
        err = capsys.readouterr().err
        # falls back to e.reason when the body isn't JSON
        assert 'Internal Server Error' in err

    def test_truncated_response(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(client_utils, 'ensure_running', lambda: 'http://127.0.0.1:1')

        class FakeResp:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *a: object) -> None:
                pass

            def read(self) -> bytes:
                raise http.client.IncompleteRead(b'', 1020)

        monkeypatch.setattr(client_utils.urllib.request, 'urlopen', lambda *a, **kw: FakeResp())
        with pytest.raises(SystemExit) as ei:
            client_utils.get_request('/api/health')
        assert ei.value.code == 1
        err = capsys.readouterr().err
        assert 'bad response from daemon' in err
        assert 'IncompleteRead' in err

    def test_url_error_unreachable(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(client_utils, 'ensure_running', lambda: 'http://127.0.0.1:1')

        def boom(*a: object, **kw: object) -> None:
            raise urllib.error.URLError('connection refused')

        monkeypatch.setattr(client_utils.urllib.request, 'urlopen', boom)
        with pytest.raises(SystemExit) as ei:
            client_utils.get_request('/api/health')
        assert ei.value.code == 1
        assert 'cannot reach daemon' in capsys.readouterr().err


class TestShell:
    """Exercise the REPL via subprocess to cover input/eof/error branches."""

    def test_shell_runs_health(self, pxt_daemon: int) -> None:
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon)}
        r = subprocess.run(
            ['pxt', 'shell'], input='health\nexit\n', capture_output=True, text=True, env=env, timeout=30, check=False
        )
        assert r.returncode == 0, r.stderr
        # the health response is JSON; should appear in stdout between two prompts
        assert '"service": "pxt"' in r.stdout

    def test_shell_eof(self, pxt_daemon: int) -> None:
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

    def test_shell_unknown_command(self, pxt_daemon: int) -> None:
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
        # bad command produces a stderr line, but the follow-up health command still runs
        assert 'unknown command' in r.stderr
        assert '"service": "pxt"' in r.stdout

    def test_shell_nested(self, pxt_daemon: int) -> None:
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
        # help lists every non-shell command
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

    def test_shell_help_commands(self, capsys: pytest.CaptureFixture) -> None:
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
    def test_write_pidfile(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = str(tmp_path / 'sub' / 'pid')
        monkeypatch.setattr(server_daemon, 'pidfile_path', lambda: path)
        server_daemon._write_pidfile()
        with open(path, encoding='utf-8') as f:
            assert int(f.read().strip()) == os.getpid()
        server_daemon._remove_pidfile_if_ours()
        assert not os.path.exists(path)

    def test_write_pidfile_overwrites_stale(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = str(tmp_path / 'pid')
        monkeypatch.setattr(server_daemon, 'pidfile_path', lambda: path)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('999999999')
        server_daemon._write_pidfile()
        with open(path, encoding='utf-8') as f:
            assert int(f.read().strip()) == os.getpid()

    def test_remove_pidfile_only_own(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = str(tmp_path / 'pid')
        monkeypatch.setattr(server_daemon, 'pidfile_path', lambda: path)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('12345')
        server_daemon._remove_pidfile_if_ours()
        assert os.path.exists(path)

    def test_remove_pidfile_missing_no_raise(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server_daemon, 'pidfile_path', lambda: str(tmp_path / 'never-existed'))
        server_daemon._remove_pidfile_if_ours()

    def test_remove_pidfile_oserror(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Locks the `except OSError: pass` branch around os.remove()."""
        path = str(tmp_path / 'pid')
        monkeypatch.setattr(server_daemon, 'pidfile_path', lambda: path)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(str(os.getpid()))

        def boom(_p: str) -> None:
            raise OSError('vanished')

        monkeypatch.setattr(server_daemon.os, 'remove', boom)
        server_daemon._remove_pidfile_if_ours()  # must not raise

    def test_main_bind_succeeds(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: bind succeeds -> write pidfile, register atexit, run server."""
        monkeypatch.setattr(server_daemon, 'pidfile_path', lambda: str(tmp_path / 'pid'))
        monkeypatch.setattr(server_daemon, 'get_port', lambda: 12345)
        fake_server = object()
        bound: list[int] = []
        ran: list[object] = []

        def fake_bind(p: int) -> object:
            bound.append(p)
            return fake_server

        monkeypatch.setattr(server_daemon, 'bind', fake_bind)
        monkeypatch.setattr(server_daemon, 'run', lambda s: ran.append(s))
        monkeypatch.setattr(server_daemon.atexit, 'register', lambda _fn: None)
        server_daemon.main([])
        assert bound == [12345]
        assert ran == [fake_server]

    def test_main_defers_to_live_peer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """bind() OSError + a live pxt daemon on the port: exit 0 silently."""
        monkeypatch.setattr(server_daemon, 'get_port', lambda: 12345)

        def fail(_p: int) -> None:
            raise OSError('address already in use')

        monkeypatch.setattr(server_daemon, 'bind', fail)
        monkeypatch.setattr(server_daemon, 'is_running', lambda: True)
        with pytest.raises(SystemExit) as info:
            server_daemon.main([])
        assert info.value.code == 0

    def test_main_reports_unrelated_port_holder(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """bind() OSError + nobody at /api/health: print the error, exit 1."""
        monkeypatch.setattr(server_daemon, 'get_port', lambda: 12345)

        def fail(_p: int) -> None:
            raise OSError('address already in use')

        monkeypatch.setattr(server_daemon, 'bind', fail)
        monkeypatch.setattr(server_daemon, 'is_running', lambda: False)
        with pytest.raises(SystemExit) as info:
            server_daemon.main([])
        assert info.value.code == 1
        captured = capsys.readouterr()
        assert 'bind to 127.0.0.1:12345 failed' in captured.err


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

    def test_dir_size_skips_stat_errors(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / 'a').write_bytes(b'x' * 10)
        real_stat = os.stat

        def flaky(p: str | os.PathLike, *, follow_symlinks: bool = True) -> object:
            if str(p).endswith('a'):
                raise OSError('vanished')
            return real_stat(p, follow_symlinks=follow_symlinks)

        with pytest.MonkeyPatch.context() as m:
            m.setattr(server_routes.os, 'stat', flaky)
            # the failing file is skipped; the walk still completes
            assert server_routes._dir_size(str(tmp_path)) == 0

    def test_redact_db_password_none(self) -> None:
        assert server_routes._redact_db_password(None) is None

    def test_redact_db_password_hidden(self) -> None:
        out = server_routes._redact_db_password('postgresql://user:secret@host/db')
        assert out is not None
        assert 'secret' not in out

    def test_redact_db_password_unparseable(self) -> None:
        # malformed URL -> caught and returns None rather than 500ing /status
        assert server_routes._redact_db_password('::: not a url :::') is None

    def test_safe_count_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeT:
            def count(self) -> int:
                raise excs.NotFoundError(excs.ErrorCode.PATH_NOT_FOUND, 'catalog gone')

        monkeypatch.setattr(server_routes.pxt, 'get_table', lambda p: FakeT())
        assert server_routes._tbl_count('any/path') is None

    def test_resolve_path_rejects_control_chars(self) -> None:
        # Defense in depth: every ASCII control character (including LF, which the
        # route-matching regex already filters out) must be rejected during path resolution
        # so future code paths that bypass the router can't smuggle them through.
        req = server_router.Request(query={}, body_bytes=b'')
        for ch in ('\n', '\r', '\x00', '\x01', '\x1f', '\x7f'):
            with pytest.raises(excs.RequestError) as ei:
                req.resolve_path(f'foo{ch}bar')
            assert 'control characters' in str(ei.value)
        # plain printable paths still pass through
        assert req.resolve_path('foo/bar') == 'foo/bar'
        assert req.resolve_path('') == ''


class TestDaemonCmd:
    """`pxt daemon start|stop|restart|status`. The action handlers in
    pixeltable_cli/client/commands/daemon.py thread through utils/client_utils helpers; tests mock those at
    the boundary so they verify the command's decision logic without spawning real daemons."""

    def test_start_prints_endpoint(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(daemon_cmd, 'ensure_running', lambda: 'http://127.0.0.1:22090')
        monkeypatch.setattr(daemon_cmd, 'fetch_health', lambda: {'pid': 4242})
        daemon_cmd.run(['start'])
        out = capsys.readouterr().out
        assert 'http://127.0.0.1:22090' in out
        assert '4242' in out

    def test_start_propagates_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        def boom() -> str:
            raise RuntimeError('cannot spawn daemon: simulated failure')

        monkeypatch.setattr(daemon_cmd, 'ensure_running', boom)
        with pytest.raises(SystemExit) as ei:
            daemon_cmd.run(['start'])
        assert ei.value.code == 1
        assert 'cannot spawn daemon' in capsys.readouterr().err

    def test_stop_pid_matches(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setattr(daemon_cmd, 'read_pidfile', lambda: 4242)
        monkeypatch.setattr(daemon_cmd, 'fetch_health', lambda: {'pid': 4242})
        monkeypatch.setattr(daemon_cmd, 'pidfile_path', lambda: str(tmp_path / 'pid'))
        killed: list[int] = []
        monkeypatch.setattr(daemon_cmd, 'kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        daemon_cmd.run(['stop'])
        assert killed == [4242]
        assert 'PID 4242' in capsys.readouterr().out

    def test_stop_no_daemon_exits_1(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(daemon_cmd, 'read_pidfile', lambda: None)
        monkeypatch.setattr(daemon_cmd, 'fetch_health', lambda: None)
        with pytest.raises(SystemExit) as ei:
            daemon_cmd.run(['stop'])
        assert ei.value.code == 1
        assert 'no daemon running' in capsys.readouterr().err

    def test_stop_pidfile_no_responder(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: pathlib.Path
    ) -> None:
        # Daemon hung or crashed: pidfile points somewhere, /health silent. kill_and_wait
        # is idempotent on a dead PID, so the kill attempt is safe either way.
        monkeypatch.setattr(daemon_cmd, 'read_pidfile', lambda: 9999)
        monkeypatch.setattr(daemon_cmd, 'fetch_health', lambda: None)
        monkeypatch.setattr(daemon_cmd, 'pidfile_path', lambda: str(tmp_path / 'pid'))
        killed: list[int] = []
        monkeypatch.setattr(daemon_cmd, 'kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        daemon_cmd.run(['stop'])
        assert killed == [9999]
        assert 'PID 9999' in capsys.readouterr().out

    def test_stop_pid_mismatch_refuses_without_force(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setattr(daemon_cmd, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(daemon_cmd, 'fetch_health', lambda: {'pid': 200})
        killed: list[int] = []
        monkeypatch.setattr(daemon_cmd, 'kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        with pytest.raises(SystemExit) as ei:
            daemon_cmd.run(['stop'])
        assert ei.value.code == 1
        assert 'does not match pidfile' in capsys.readouterr().err
        assert killed == []

    def test_stop_pid_mismatch_force_kills_responder(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setattr(daemon_cmd, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(daemon_cmd, 'fetch_health', lambda: {'pid': 200})
        monkeypatch.setattr(daemon_cmd, 'pidfile_path', lambda: str(tmp_path / 'pid'))
        killed: list[int] = []
        monkeypatch.setattr(daemon_cmd, 'kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        daemon_cmd.run(['stop', '--force'])
        # --force on mismatch kills the responder, not the tracked pidfile PID
        assert killed == [200]

    def test_stop_no_pidfile_refuses(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(daemon_cmd, 'read_pidfile', lambda: None)
        monkeypatch.setattr(daemon_cmd, 'fetch_health', lambda: {'pid': 200})
        killed: list[int] = []
        monkeypatch.setattr(daemon_cmd, 'kill_and_wait', lambda pid, timeout=5.0: killed.append(pid))
        with pytest.raises(SystemExit) as ei:
            daemon_cmd.run(['stop'])
        assert ei.value.code == 1
        assert 'no pidfile' in capsys.readouterr().err
        assert killed == []

    def test_status_prints_identity_text(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(
            daemon_cmd,
            'fetch_health',
            lambda timeout=None: {
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

    def test_status_json(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        payload = {'pid': 4242, 'service': 'pxt', 'pixeltable_env': {}}
        monkeypatch.setattr(daemon_cmd, 'fetch_health', lambda timeout=None: payload)
        daemon_cmd.run(['status', '--json'])
        assert json.loads(capsys.readouterr().out) == payload

    def test_status_no_daemon_exits_1(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(daemon_cmd, 'fetch_health', lambda timeout=None: None)
        monkeypatch.setattr(daemon_cmd, 'port_is_open', lambda: False)
        with pytest.raises(SystemExit) as ei:
            daemon_cmd.run(['status'])
        assert ei.value.code == 1
        assert 'no daemon running' in capsys.readouterr().err

    def test_status_busy_daemon_exits_1(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        """A daemon that holds the port but does not answer is reported as busy, not as absent."""
        monkeypatch.setattr(daemon_cmd, 'fetch_health', lambda timeout=None: None)
        monkeypatch.setattr(daemon_cmd, 'port_is_open', lambda: True)
        with pytest.raises(SystemExit) as ei:
            daemon_cmd.run(['status'])
        assert ei.value.code == 1
        err = capsys.readouterr().err
        assert 'busy with a request' in err
        assert 'no daemon running' not in err

    def test_restart(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: pathlib.Path
    ) -> None:
        # First cycle: daemon present with matching pids; stop kills it, then start spawns
        # a new one. fetch_health() returns a daemon for stop, then a daemon (different PID)
        # after start.
        states = iter(
            [
                {'pid': 100},  # for the stop branch
                {'pid': 200},  # for the start branch's post-spawn lookup
            ]
        )
        monkeypatch.setattr(daemon_cmd, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(daemon_cmd, 'fetch_health', lambda: next(states))
        monkeypatch.setattr(daemon_cmd, 'pidfile_path', lambda: str(tmp_path / 'pid'))
        actions: list[str] = []
        monkeypatch.setattr(daemon_cmd, 'kill_and_wait', lambda pid, timeout=5.0: actions.append(f'kill:{pid}'))

        def fake_ensure_running() -> str:
            actions.append('spawn')
            return 'http://127.0.0.1:22090'

        monkeypatch.setattr(daemon_cmd, 'ensure_running', fake_ensure_running)

        daemon_cmd.run(['restart'])
        assert actions == ['kill:100', 'spawn']
        out = capsys.readouterr().out
        assert 'http://127.0.0.1:22090' in out
        assert '200' in out

    def test_restart_no_daemon(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: pathlib.Path
    ) -> None:
        # Nothing to stop initially; restart should still proceed to start without erroring.
        states = iter([None, {'pid': 200}])
        monkeypatch.setattr(daemon_cmd, 'read_pidfile', lambda: None)
        monkeypatch.setattr(daemon_cmd, 'fetch_health', lambda: next(states))
        monkeypatch.setattr(daemon_cmd, 'ensure_running', lambda: 'http://127.0.0.1:22090')
        monkeypatch.setattr(daemon_cmd, 'pidfile_path', lambda: str(tmp_path / 'pid'))
        daemon_cmd.run(['restart'])
        out = capsys.readouterr().out
        assert 'http://127.0.0.1:22090' in out


class TestPxtPathValidator:
    """Pydantic validator that backs MoveBody.path / new_path."""

    def test_accepts_empty(self) -> None:
        from pixeltable_cli.models import _validate_pxt_path

        assert _validate_pxt_path(None) is None
        assert _validate_pxt_path('') == ''

    def test_accepts_valid_path(self) -> None:
        from pixeltable_cli.models import MoveBody

        m = MoveBody(path='a/b', new_path='c')
        assert m.path == 'a/b'
        assert m.new_path == 'c'

    def test_rejects_bad_shape(self) -> None:
        import pydantic

        from pixeltable_cli.models import MoveBody

        with pytest.raises(pydantic.ValidationError):
            MoveBody(path='a.b', new_path='c')
        with pytest.raises(pydantic.ValidationError):
            MoveBody(path='a//b', new_path='c')
        with pytest.raises(pydantic.ValidationError):
            MoveBody(path='a/b', new_path='trailing/')


class TestDashboardCommand:
    """`pxt dashboard` URL launcher, in-process."""

    def test_ensure_running_failure_exits(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        from pixeltable_cli.client.commands import dashboard as dashboard_cmd

        def boom() -> None:
            raise RuntimeError('cannot reach daemon')

        monkeypatch.setattr(dashboard_cmd, 'ensure_running', boom)
        with pytest.raises(SystemExit) as info:
            dashboard_cmd.run([])
        assert info.value.code == 1
        assert 'cannot reach daemon' in capsys.readouterr().err


class TestBadPathArgRejection:
    """Each path-taking command rejects malformed paths client-side with argparse exit 2."""

    def _assert_arg_error(
        self, runner: Callable[[list[str]], None], argv: list[str], capsys: pytest.CaptureFixture
    ) -> None:
        with pytest.raises(SystemExit) as info:
            runner(argv)
        assert info.value.code == 2
        assert 'pxt paths' in capsys.readouterr().err

    def test_columns_bad_path(self, capsys: pytest.CaptureFixture) -> None:
        from pixeltable_cli.client.commands import columns as columns_cmd

        self._assert_arg_error(columns_cmd.run, ['a.b'], capsys)

    def test_computed_bad_path(self, capsys: pytest.CaptureFixture) -> None:
        from pixeltable_cli.client.commands import computed as computed_cmd

        self._assert_arg_error(computed_cmd.run, ['a.b'], capsys)

    def test_idxs_bad_path(self, capsys: pytest.CaptureFixture) -> None:
        from pixeltable_cli.client.commands import idxs as idxs_cmd

        self._assert_arg_error(idxs_cmd.run, ['a.b'], capsys)

    def test_mv_bad_source_path(self, capsys: pytest.CaptureFixture) -> None:
        from pixeltable_cli.client.commands import mv as mv_cmd

        self._assert_arg_error(mv_cmd.run, ['a.b', 'dst'], capsys)

    def test_mv_bad_new_dir(self, capsys: pytest.CaptureFixture) -> None:
        from pixeltable_cli.client.commands import mv as mv_cmd

        self._assert_arg_error(mv_cmd.run, ['src/foo', 'has..dot'], capsys)

    def test_rename_bad_path(self, capsys: pytest.CaptureFixture) -> None:
        from pixeltable_cli.client.commands import rename as rename_cmd

        self._assert_arg_error(rename_cmd.run, ['a.b', 'newname'], capsys)


class TestIdxsEmbeddingDisplay:
    """`pxt idxs` extra-column rendering for embedding indexes."""

    def test_embedding_extra_fields(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        from pixeltable_cli.client.commands import idxs as idxs_cmd

        resp = {
            'entries': [
                {
                    'table': 'my.tbl',
                    'name': 'idx0',
                    'index_type': 'embedding',
                    'columns': ['text'],
                    'metric': 'cosine',
                    'embedding': 'sbert',
                }
            ]
        }
        monkeypatch.setattr(idxs_cmd, 'get_request', lambda *a, **kw: resp)
        idxs_cmd.run([])
        out = capsys.readouterr().out
        assert 'cosine' in out
        assert 'sbert' in out


class TestHardeningHeaders:
    """Daemon responses carry baseline security headers (X-Content-Type-Options etc.)."""

    def test_health_hardening_headers(self, pxt_daemon: int) -> None:
        with urllib.request.urlopen(f'http://127.0.0.1:{pxt_daemon}/api/health', timeout=5) as r:
            assert r.headers.get('X-Content-Type-Options') == 'nosniff'
            assert r.headers.get('X-Frame-Options') == 'DENY'
            assert r.headers.get('Referrer-Policy') == 'no-referrer'


class TestConfigRoute:
    def test_config_route_options(self, init_env: None) -> None:
        """/api/config reports each option in KNOWN_CONFIG_OPTIONS, whatever its declared type.

        A declared type is coerced onto the configured value, which a parametric generic (eg list[X]) does
        not survive: calling it raises TypeError. The route collapses such a type to its origin first.
        """
        # in-process call into the route handler; doesn't require the daemon subprocess
        from pixeltable.config import KNOWN_CONFIG_OPTIONS
        from pixeltable_cli.server.router import Request

        resp = server_routes.config(Request(query={}, body_bytes=b''))
        reported = {(e.section, e.key) for e in resp.entries}
        expected = {(section, key) for section, options in KNOWN_CONFIG_OPTIONS.items() for key in options}
        assert reported == expected

    def test_config_route_redacts_otel_headers(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        from pixeltable_cli.server.router import Request

        monkeypatch.setenv('OTEL_EXPORTER_OTLP_HEADERS', 'Authorization=Bearer top-secret')
        resp = server_routes.config(Request(query={}, body_bytes=b''))
        headers = [e for e in resp.entries if e.section == 'otel' and e.key == 'exporter_otlp_headers']
        assert len(headers) == 1
        assert headers[0].value == '<redacted>'


class TestPerPortPaths:
    """Pidfile and log paths must be parameterized by PXT_PORT so that daemons running on
    different ports don't share state."""

    def test_log_path_includes_port(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path))
        monkeypatch.setenv('PXT_PORT', '12345')
        p1 = client_utils._daemon_log_path()
        monkeypatch.setenv('PXT_PORT', '54321')
        p2 = client_utils._daemon_log_path()
        assert p1 != p2, f'log path collides across ports: {p1} == {p2}'
        assert '12345' in p1
        assert '54321' in p2


class TestHostedCommandHelp:
    """Subcommands and options offered by `pxt db`, `pxt service` and `pxt org`."""

    @pytest.mark.parametrize(
        ('module', 'argv', 'expected'),
        [
            (db_cmd, ['--help'], ['diff', 'update', 'create', 'list', 'build-image', 'status']),
            (db_cmd, ['update', '--help'], ['--allow-destructive', '--dry-run']),
            (org_cmd, ['--help'], ['list', 'status']),
        ],
    )
    def test_help(
        self, module: ModuleType, argv: list[str], expected: list[str], capsys: pytest.CaptureFixture
    ) -> None:
        with pytest.raises(SystemExit) as info:
            module.run(argv)
        assert info.value.code == 0
        out = capsys.readouterr().out
        assert all(token in out for token in expected), out


def _forwarded_request(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[server_router.Request], Any],
    *,
    body: dict[str, Any] | None = None,
    query: dict[str, list[str]] | None = None,
) -> Any:
    """Call a daemon route with the given body/query and return the management API request it sent."""
    sent: list[Any] = []

    def api_call(request: Any) -> dict[str, Any]:
        sent.append(request)
        return {}

    monkeypatch.setattr(management_client, 'api_call', api_call)
    req = server_router.Request(
        query={} if query is None else query, body_bytes=b'' if body is None else json.dumps(body).encode()
    )
    handler(req)
    assert len(sent) == 1, sent
    return sent[0]


_POST_ROUTE_REQUESTS = [
    (
        server_routes.create_db,
        {'org': 'acme', 'db': 'main', 'location': 'aws', 'region': 'us-east-1'},
        CreateDbRequest(org='acme', db='main', location='aws', region='us-east-1'),
    ),
    (server_routes.delete_db, {'org': 'acme', 'db': 'main'}, DeleteDbRequest(org='acme', db='main')),
    (server_routes.start_db, {'org': 'acme', 'db': 'main'}, StartDbRequest(org='acme', db='main')),
    (server_routes.stop_db, {'org': 'acme', 'db': 'main'}, StopDbRequest(org='acme', db='main')),
]

_GET_ROUTE_REQUESTS = [
    (server_routes.list_orgs, {}, ListOrgsRequest()),
    (server_routes.list_dbs, {'org': ['acme']}, ListDbRequest(org='acme')),
    (server_routes.get_db, {'org': ['acme'], 'db': ['main']}, GetDbRequest(org='acme', db='main')),
]


class TestCloudRouteRequests:
    """The management API request each cloud route builds from its body or query string."""

    @pytest.mark.parametrize(('handler', 'body', 'expected'), _POST_ROUTE_REQUESTS)
    def test_post_route(
        self,
        handler: Callable[[server_router.Request], Any],
        body: dict[str, Any],
        expected: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert _forwarded_request(monkeypatch, handler, body=body) == expected

    @pytest.mark.parametrize(('handler', 'query', 'expected'), _GET_ROUTE_REQUESTS)
    def test_get_route(
        self,
        handler: Callable[[server_router.Request], Any],
        query: dict[str, list[str]],
        expected: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert _forwarded_request(monkeypatch, handler, query=query) == expected

    def test_get_org_picks_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orgs = {'orgs': [{'org': 'other', 'org_id': 'o0'}, {'org': 'acme', 'org_id': 'o1'}]}
        monkeypatch.setattr(management_client, 'api_call', lambda request: orgs)
        assert server_routes.get_org(server_router.Request(query={'org': ['acme']}, body_bytes=b'')) == {
            'org': {'org': 'acme', 'org_id': 'o1'}
        }
        with pytest.raises(excs.NotFoundError, match="Org 'nope' not found"):
            server_routes.get_org(server_router.Request(query={'org': ['nope']}, body_bytes=b''))


class TestHostedCommandRequests:
    """The bodies `pxt db` and `pxt service` post, as the cloud routes read them."""

    def _posted_body(self, monkeypatch: pytest.MonkeyPatch, module: ModuleType, argv: list[str]) -> dict[str, Any]:
        """Run one CLI command with the daemon stubbed out, and return the body it posted."""
        posted: list[dict[str, Any]] = []

        # each command checks that its resource reached the state the operation aims for, so the stubs
        # report the state that operation ends in
        def post_request(path: str, body: dict[str, Any]) -> dict[str, Any]:
            posted.append(body)
            return {'state': 'STOPPED' if path.endswith('/stop') else 'AVAILABLE'}

        def poll(org: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
            pending = next((a for a in args if isinstance(a, set)), set())
            return {'state': 'STOPPED' if 'STOPPING' in pending else 'AVAILABLE'}

        monkeypatch.setattr(module, 'post_request', post_request)
        monkeypatch.setattr(module, 'get_request', lambda path, params=None: {})
        monkeypatch.setattr(module, 'poll_db', poll, raising=False)
        module.run(argv)
        assert len(posted) == 1, posted
        return posted[0]

    @pytest.mark.parametrize(
        ('module', 'argv', 'handler', 'expected'),
        [
            (
                db_cmd,
                ['create', 'pxt://acme:main'],
                server_routes.create_db,
                CreateDbRequest(org='acme', db='main', location='aws', region='us-east-1'),
            ),
            (db_cmd, ['start', 'pxt://acme:main'], server_routes.start_db, StartDbRequest(org='acme', db='main')),
            (db_cmd, ['stop', 'pxt://acme:main'], server_routes.stop_db, StopDbRequest(org='acme', db='main')),
            (db_cmd, ['delete', 'pxt://acme:main'], server_routes.delete_db, DeleteDbRequest(org='acme', db='main')),
        ],
    )
    def test_command_body(
        self,
        module: ModuleType,
        argv: list[str],
        handler: Callable[[server_router.Request], Any],
        expected: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        body = self._posted_body(monkeypatch, module, argv)
        assert _forwarded_request(monkeypatch, handler, body=body) == expected

    @pytest.mark.parametrize('db', ['main', 'my-db', 'db1', 'video-search', 'a' * 29])
    def test_create_db_accepts_valid_name(self, db: str) -> None:
        assert CreateDbRequest(org='acme', db=db).db == db

    @pytest.mark.parametrize('db', ['My_DB', 'a_b', 'ACME', 'db-', '-db', 'a' * 30, 'my db', 'my.db', 'main\n', ''])
    def test_create_db_rejects_invalid_name(self, db: str) -> None:
        with pytest.raises(pydantic.ValidationError):
            CreateDbRequest(org='acme', db=db)


class TestServiceOtel:
    @pytest.mark.otel
    def test_run_otel(self, tmp_path: pathlib.Path) -> None:
        """--otel resolves the instrumentation package and wires init()/instrument_fastapi() into run."""
        skip_test_if_not_installed('fastapi', 'uvicorn', 'opentelemetry.instrumentation.pixeltable')
        app_file = tmp_path / 'app.py'
        app_file.write_text('', encoding='utf-8')  # the app it declares is patched out below
        app = SimpleNamespace(routes=[])  # stands in for the FastAPI app, which run() counts the routes of

        with (
            patch('pixeltable.serving._app.create_app', return_value=(app, {})),
            patch('uvicorn.run') as mock_run,
            patch('opentelemetry.instrumentation.pixeltable.init') as mock_otel_init,
            patch('opentelemetry.instrumentation.pixeltable.instrument_fastapi') as mock_instrument_fastapi,
        ):
            service_cmd.run(['run', str(app_file), 'my_dir', 'ingest', '--otel'])
        mock_otel_init.assert_called_once_with()
        mock_instrument_fastapi.assert_called_once_with(app)
        mock_run.assert_called_once()


def _declared_spec(app_file: str, name: str) -> Any:
    """The spec of one of a file's services, as a manager computes it before starting the service."""
    module = load_app_module(app_file, subject='application file')
    return service_spec(name, services_by_name(module, app_file)[name], module_routers(module))


class TestHostedServiceManager:
    """What ServiceManagerProxy asks the management API for, and what it makes of the answers."""

    @pytest.fixture(autouse=True)
    def serving_installed(self) -> None:
        """Skip where fastapi is absent: the corpus files these tests start declare FastAPIRouter services."""
        skip_test_if_not_installed('fastapi')

    @pytest.fixture
    def api(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """A management API holding instances in a dict, recording every request it is sent."""

        class Api:
            instances: dict[str, dict[str, Any]]
            sent: list[Any]

            def __init__(self) -> None:
                self.instances = {}
                self.sent = []

            def add(self, name: str, *, base_path: str = '', state: str = 'AVAILABLE', **fields: Any) -> None:
                self.instances[name] = {
                    'service_name': name,
                    'base_path': base_path,
                    'endpoint': f'https://acme-main.pixeltable.com/{name}',
                    'app_module': 'apps.basic',
                    'spec': {'name': name, 'routes': [], 'app_paths': []},
                    'state': state,
                    # every instance reports the project it loaded
                    'fingerprint': project_fingerprint(Config.get().project_root, None).model_dump(),
                    **fields,
                }

            def __call__(self, request: Any) -> dict[str, Any]:
                self.sent.append(request)
                op = request.operation_type.value
                if op == 'list_service_instances':
                    return {'instances': list(self.instances.values())}
                if op == 'create_service_instance':
                    self.add(
                        request.service_name,
                        base_path=request.base_path,
                        app_module=request.app_module,
                        spec=request.spec,
                    )
                elif op == 'update_service_instance':
                    self.instances[request.service_name].update(app_module=request.app_module, spec=request.spec)
                elif op == 'start_service_instance':
                    instance = self.instances[request.service_name]
                    # an instance whose failure is recorded stays FAILED, as one that cannot be brought up does
                    instance['state'] = 'FAILED' if instance.get('error') is not None else 'AVAILABLE'
                elif op == 'stop_service_instance':
                    self.instances[request.service_name]['state'] = 'STOPPED'
                elif op == 'delete_service_instance':
                    del self.instances[request.service_name]
                return {'instance': self.instances.get(request.service_name, {})}

        api = Api()
        monkeypatch.setattr(management_client, 'api_call', api)
        return api

    def _manager(self) -> Any:
        from pixeltable.serving.service_manager import get_manager

        return get_manager('pxt://acme:main/app')

    def test_start_absent(self, api: Any, apps: Callable[[str], str]) -> None:
        instance = self._manager().start(apps('basic.py'), 'ingest', 'app')
        assert [r.operation_type.value for r in api.sent] == [
            'list_service_instances',
            'create_service_instance',
            'list_service_instances',
        ]
        created = api.sent[1]
        assert (created.org, created.db, created.service_name, created.base_path) == ('acme', 'main', 'ingest', 'app')
        assert created.app_module == 'apps.basic'
        assert created.spec.name == 'ingest'
        assert instance.state is ServiceInstanceState.AVAILABLE

    def test_start_changed(self, api: Any, apps: Callable[[str], str]) -> None:
        api.add('ingest', base_path='app', spec={'name': 'ingest', 'routes': [], 'app_paths': []})
        self._manager().start(apps('basic.py'), 'ingest', 'app')
        assert [r.operation_type.value for r in api.sent].count('update_service_instance') == 1

    def test_start_stopped(self, api: Any, apps: Callable[[str], str]) -> None:
        spec = _declared_spec(apps('basic.py'), 'ingest')
        api.add('ingest', base_path='app', state='STOPPED', spec=spec)
        instance = self._manager().start(apps('basic.py'), 'ingest', 'app')
        assert 'start_service_instance' in [r.operation_type.value for r in api.sent]
        assert instance.state is ServiceInstanceState.AVAILABLE

    def test_start_agreeing(self, api: Any, apps: Callable[[str], str]) -> None:
        spec = _declared_spec(apps('basic.py'), 'ingest')
        api.add('ingest', base_path='app', spec=spec)
        self._manager().start(apps('basic.py'), 'ingest', 'app')
        assert [r.operation_type.value for r in api.sent] == ['list_service_instances']

    def test_start_failure(self, api: Any, apps: Callable[[str], str]) -> None:
        api.add('ingest', base_path='app', state='FAILED', error='the image has no module apps.basic')
        with pytest.raises(excs.Error, match=r'did not start; it is FAILED: the image has no module apps\.basic'):
            self._manager().start(apps('basic.py'), 'ingest', 'app')

    def test_list_by_path(self, api: Any) -> None:
        api.add('ingest', base_path='app')
        api.add('search', base_path='app/sub')
        api.add('other', base_path='elsewhere')
        manager = self._manager()
        assert [i.service_name for i in manager.list('app')] == ['ingest']
        assert sorted(i.service_name for i in manager.list('app', recursive=True)) == ['ingest', 'search']
        assert manager.get('search', 'app') is None
        assert manager.get('search', 'app/sub') is not None

    def test_stop(self, api: Any) -> None:
        api.add('ingest', base_path='app')
        manager = self._manager()
        instance = manager.get('ingest', 'app')
        assert instance is not None
        instance.stop()
        assert api.instances['ingest']['state'] == 'STOPPED'
        assert manager.get('ingest', 'app') is not None

    def test_delete(self, api: Any) -> None:
        api.add('ingest', base_path='app')
        manager = self._manager()
        instance = manager.get('ingest', 'app')
        assert instance is not None
        instance.delete()
        assert manager.get('ingest', 'app') is None


class TestHostedDatabase:
    """What `pxt db diff` compares, and what `pxt db update` applies in what order."""

    @pytest.fixture
    def api(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """A management API holding one database and its secrets, recording every request it is sent."""

        class Api:
            database: dict[str, Any] | None
            secrets: dict[str, str]
            sent: list[Any]

            def __init__(self) -> None:
                self.database = {'state': 'AVAILABLE', 'cpu': 0.5, 'memory_mb': 512, 'disk_gb': 10, 'workers': []}
                self.secrets = {}
                self.sent = []

            def __call__(self, request: Any) -> dict[str, Any]:
                self.sent.append(request)
                op = request.operation_type.value
                if op == 'get_db':
                    if self.database is None:
                        raise excs.ExternalServiceError(
                            excs.ErrorCode.PROVIDER_ERROR, 'Management API error 404', status_code=404
                        )
                    return {'database': self.database}
                if op == 'list_secrets':
                    return {'keys': sorted(self.secrets)}
                if op == 'set_secret':
                    self.secrets[request.key] = request.value
                elif op == 'delete_secret':
                    del self.secrets[request.key]
                elif op in ('build_image', 'set_project'):
                    pass
                elif op == 'create_db':
                    self.database = {'state': 'AVAILABLE', 'cpu': request.cpu, 'memory_mb': request.memory_mb}
                elif op == 'update_db':
                    assert self.database is not None
                    for field in ('cpu', 'memory_mb', 'disk_gb'):
                        if getattr(request, field) is not None:
                            self.database[field] = getattr(request, field)
                return {}

        api = Api()
        monkeypatch.setattr(management_client, 'api_call', api)
        return api

    @pytest.fixture
    def uploaded(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Records the key of each stored archive, in place of packaging and uploading one."""
        keys: list[str] = []

        def upload(config: Any, db_path: PxtPath, *, show_progress: bool = False) -> str:
            keys.append(f'{db_path.org}/{db_path.db}/project.tar.bz2')
            return keys[-1]

        monkeypatch.setattr(db, '_upload_project_archive', upload)
        return keys

    def _project(self, tmp_path: pathlib.Path, entry: str) -> None:
        """Declare one hosted database in a project at tmp_path, and make it this process's project."""
        (tmp_path / 'app.py').write_text('import pixeltable as pxt\n')
        (tmp_path / 'uv.lock').write_text('version = 1\n')
        (tmp_path / 'pixeltable.toml').write_text(f'[[pixeltable.database]]\nname = "pxt://acme:main"\n{entry}')
        Config.init(reinit=True, project_root=tmp_path)

    def _fingerprint(self, tmp_path: pathlib.Path) -> dict[str, Any]:
        """The project's fingerprint, in the form GET_DB reports it."""
        entry = Config.get().get_database_config(PxtPath.parse('pxt://acme:main', allow_empty_path=True))
        return project_fingerprint(tmp_path, entry).model_dump()

    def test_diff_capacity_secrets_placement(self, api: Any, tmp_path: pathlib.Path) -> None:
        api.secrets['stale_key'] = 'x'
        api.database['location'] = 'aws/us-east-1'
        self._project(
            tmp_path,
            'cpu = 2.0\nmemory_mb = 256\nworkers = 2\nlocation = "gcp/us-central1"\n'
            '[pixeltable.database.secrets]\nopenai_api_key = "env:OPENAI_API_KEY"\n',
        )
        api.database['fingerprint'] = self._fingerprint(tmp_path)
        plan = db.db_diff('pxt://acme:main')

        assert [(op.target, op.name, op.severity) for op in plan.ops] == [
            ('capacity', 'cpu', 'additive'),
            ('capacity', 'memory_mb', 'destructive'),
            ('capacity', 'workers', 'additive'),
            ('secret', 'openai_api_key', 'additive'),
            ('secret', 'stale_key', 'destructive'),
            ('placement', 'location', 'unsupported'),
        ]
        assert plan.resolution == 'unsupported'
        assert plan.destructive

    def test_diff_agreement(self, api: Any, tmp_path: pathlib.Path) -> None:
        self._project(tmp_path, 'cpu = 0.5\nmemory_mb = 512\ndisk_gb = 10\n')
        api.database['fingerprint'] = self._fingerprint(tmp_path)
        plan = db.db_diff('pxt://acme:main')
        assert plan.ops == []
        assert plan.in_agreement
        assert plan.resolution == 'up_to_date'

    def test_diff_without_project(self, api: Any, tmp_path: pathlib.Path) -> None:
        """A database that reports no fingerprint is missing both artifacts, whatever else agrees."""
        self._project(tmp_path, 'cpu = 0.5\nmemory_mb = 512\ndisk_gb = 10\n')
        plan = db.db_diff('pxt://acme:main')
        assert [(op.target, op.name) for op in plan.ops] == [('image', 'image'), ('archive', 'project')]
        assert plan.summary.rebuild

    def test_diff_project_vs_image(self, api: Any, tmp_path: pathlib.Path) -> None:
        self._project(tmp_path, '')
        api.database['fingerprint'] = self._fingerprint(tmp_path)

        # a source edit sends the project again; the environment the image holds is untouched
        (tmp_path / 'app.py').write_text('import pixeltable as pxt  # edited\n')
        plan = db.db_diff('pxt://acme:main')
        assert [op.target for op in plan.ops] == ['archive']
        assert plan.ops[0].description == 'the project will be sent to the database again: app.py changed'
        assert not plan.summary.rebuild

        # a lockfile edit is both: the environment moved, and so did a file the archive holds
        (tmp_path / 'uv.lock').write_text('version = 2\n')
        plan = db.db_diff('pxt://acme:main')
        assert [op.target for op in plan.ops] == ['image', 'archive']
        assert plan.ops[0].description == 'the image will be rebuilt: uv.lock changed'
        assert plan.summary.rebuild

    def test_diff_absent_database(self, api: Any, tmp_path: pathlib.Path) -> None:
        api.database = None
        self._project(tmp_path, 'cpu = 2.0\n')
        plan = db.db_diff('pxt://acme:main')
        assert plan.resolution == 'create'
        assert not plan.exists
        assert plan.state is None
        # a create subsumes the operations that constitute it
        assert plan.ops == []

    def test_update_order(
        self, api: Any, uploaded: list[str], tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
        self._project(
            tmp_path,
            'cpu = 2.0\nsystem_dependencies = ["ffmpeg"]\n'
            '[pixeltable.database.secrets]\nopenai_api_key = "env:OPENAI_API_KEY"\n',
        )
        api.database['fingerprint'] = self._fingerprint(tmp_path)
        (tmp_path / 'app.py').write_text('import pixeltable as pxt  # edited\n')

        plan = db.db_update('pxt://acme:main')
        assert [r.operation_type.value for r in api.sent if r.operation_type.value != 'get_db'] == [
            'list_secrets',
            'set_secret',
            'set_project',
            'update_db',
        ]
        assert api.secrets == {'openai_api_key': 'sk-test'}
        # one archive serves both artifacts, so it is stored once
        assert uploaded == ['acme/main/project.tar.bz2']
        assert api.database['cpu'] == 2.0
        assert all(op.status == 'applied' for op in plan.ops)
        assert plan.status == 'applied'

    def test_update_image(self, api: Any, uploaded: list[str], tmp_path: pathlib.Path) -> None:
        self._project(tmp_path, 'system_dependencies = ["ffmpeg"]\n')
        api.database['fingerprint'] = self._fingerprint(tmp_path)
        (tmp_path / 'uv.lock').write_text('version = 2\n')

        db.db_update('pxt://acme:main')
        build = next(r for r in api.sent if r.operation_type.value == 'build_image')
        entry = Config.get().get_database_config(PxtPath.parse('pxt://acme:main', allow_empty_path=True))
        assert build.project_key == 'acme/main/project.tar.bz2'
        assert build.system_dependencies == ['ffmpeg']
        assert build.python_version == project_fingerprint(tmp_path, entry).python_version
        assert build.image_digest == project_fingerprint(tmp_path, entry).image_digest()
        assert build.pxt_md_version == metadata.VERSION

    def test_update_absent_database(self, api: Any, uploaded: list[str], tmp_path: pathlib.Path) -> None:
        api.database = None
        self._project(tmp_path, 'cpu = 2.0\nworkers = 2\nlocation = "aws"\nregion = "us-east-1"\n')
        db.db_update('pxt://acme:main')
        created = next(r for r in api.sent if r.operation_type.value == 'create_db')
        assert (created.org, created.db, created.location, created.region) == ('acme', 'main', 'aws', 'us-east-1')
        assert (created.cpu, created.workers) == (2.0, 2)
        # a created database reports no fingerprint, so it is given both artifacts
        assert [r.operation_type.value for r in api.sent].count('build_image') == 1
        assert uploaded == ['acme/main/project.tar.bz2']

    def test_update_refuses_shrink(self, api: Any, tmp_path: pathlib.Path) -> None:
        self._project(tmp_path, 'memory_mb = 256\n')
        api.database['fingerprint'] = self._fingerprint(tmp_path)
        with pytest.raises(excs.Error, match=r'(?s)destructive changes: memory_mb.*--allow-destructive'):
            db.db_update('pxt://acme:main')
        assert api.database['memory_mb'] == 512

        db.db_update('pxt://acme:main', allow_destructive=True)
        assert api.database['memory_mb'] == 256

    def test_secret_binding(self, api: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        self._project(tmp_path, '[pixeltable.database.secrets]\nopenai_api_key = "env:OPENAI_API_KEY"\n')
        with pytest.raises(excs.Error, match='OPENAI_API_KEY, which is not set in the environment'):
            db.db_update('pxt://acme:main')

        self._project(tmp_path, '[pixeltable.database.secrets]\nopenai_api_key = "sk-in-the-file"\n')
        with pytest.raises(excs.Error, match="write 'env:NAME' to name the environment variable"):
            db.db_update('pxt://acme:main')
        assert api.secrets == {}

    def test_errors(self, api: Any, tmp_path: pathlib.Path) -> None:
        self._project(tmp_path, 'cpu = 2.0\n')
        with pytest.raises(excs.Error, match='does not name a hosted database'):
            db.db_diff('my_dir')

        with pytest.raises(excs.Error, match=r'no \[\[pixeltable.database\]\] entry names'):
            db.db_diff('pxt://acme:other')


class TestHostedUriHelpers:
    """URI parsing / printing helpers shared by the hosted-CLI commands."""

    def test_split_pxt_uri(self) -> None:
        assert utils.split_pxt_uri('pxt://acme') == ('acme', None, None)
        assert utils.split_pxt_uri('pxt://acme:main') == ('acme', 'main', None)
        assert utils.split_pxt_uri('pxt://acme:main/services/foo') == ('acme', 'main', 'services/foo')
        assert utils.split_pxt_uri('pxt://acme:main/') == ('acme', 'main', '')
        assert utils.split_pxt_uri('not-a-uri') is None

    def test_parse_db_uri(self) -> None:
        assert hosted.parse_db_uri('pxt://acme:main') == ('acme', 'main')

    @pytest.mark.parametrize('bad', ['pxt://acme', 'pxt://acme:main/tbl', 'pxt://acme:main/', 'nope'])
    def test_parse_db_uri_rejects(self, bad: str, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as info:
            hosted.parse_db_uri(bad)
        assert info.value.code == 2
        assert 'pxt://org:db' in capsys.readouterr().err

    def test_parse_org_uri(self) -> None:
        assert hosted.parse_org_uri('pxt://acme') == 'acme'

    @pytest.mark.parametrize('bad', ['pxt://acme:main', 'pxt://acme:main/x', 'pxt://acme/', 'nope'])
    def test_parse_org_uri_rejects(self, bad: str) -> None:
        with pytest.raises(SystemExit) as info:
            hosted.parse_org_uri(bad)
        assert info.value.code == 2

    def test_parse_base_uri(self) -> None:
        assert hosted.parse_base_uri('pxt://acme:main') == ('acme', 'main', '')
        assert hosted.parse_base_uri('pxt://acme:main/dir/sub') == ('acme', 'main', 'dir/sub')

    @pytest.mark.parametrize('bad', ['pxt://acme', 'nope'])
    def test_parse_base_uri_rejects(self, bad: str) -> None:
        with pytest.raises(SystemExit) as info:
            hosted.parse_base_uri(bad)
        assert info.value.code == 2

    def test_parse_service_uri(self) -> None:
        assert hosted.parse_service_uri('pxt://acme:main/services/foo') == ('acme', 'main', 'foo')

    @pytest.mark.parametrize(
        'bad',
        [
            'pxt://acme:main/tables/foo',
            'pxt://acme:main/services/',
            'pxt://acme:main/services/foo/bar',  # extra path component rejected
            'pxt://acme:main',
            'pxt://acme',
        ],
    )
    def test_parse_service_uri_rejects(self, bad: str) -> None:
        with pytest.raises(SystemExit) as info:
            hosted.parse_service_uri(bad)
        assert info.value.code == 2

    @pytest.mark.parametrize(
        ('age_s', 'expected'),
        [(0, '0s'), (45, '45s'), (90, '1m'), (3600, '1h'), (3660, '1h1m'), (86400, '1d'), (90000, '1d1h')],
    )
    def test_fmt_age(self, age_s: int, expected: str) -> None:
        assert hosted._fmt_age(age_s) == expected

    def test_print_org(self, capsys: pytest.CaptureFixture) -> None:
        hosted.print_org({'org': 'acme', 'org_id': 'o1', 'default_db': 'main'})
        out = capsys.readouterr().out
        assert 'acme' in out and 'id=o1' in out and 'default_db=main' in out

    def test_print_db(self, capsys: pytest.CaptureFixture) -> None:
        hosted.print_db({'db': 'main', 'state': 'AVAILABLE', 'location': 'aws', 'region': 'us-east-1'})
        out = capsys.readouterr().out
        assert 'main' in out and 'state=AVAILABLE' in out and 'aws/us-east-1' in out

    def test_print_service_prints_routes(self, capsys: pytest.CaptureFixture) -> None:
        hosted.print_service(
            {
                'service_name': 'svc',
                'state': 'AVAILABLE',
                'base_path': 'main',
                'workers_min': 1,
                'endpoint': 'https://svc.example',
                'service_config': json.dumps({'prefix': '/v1', 'routes': [{'method': 'post', 'path': '/insert'}]}),
            }
        )
        out = capsys.readouterr().out
        assert 'svc' in out and 'state=AVAILABLE' in out
        assert 'POST  https://svc.example/v1/insert' in out

    def test_print_workers(self, capsys: pytest.CaptureFixture) -> None:
        hosted._print_workers(
            [{'pod_id': 'pod-1', 'status': 'Running', 'ready': 1, 'total': 1, 'restarts': 0, 'age_s': 45}]
        )
        out = capsys.readouterr().out
        assert 'POD ID' in out and 'pod-1' in out and 'Running' in out
        hosted._print_workers([])  # empty prints nothing
        assert capsys.readouterr().out == ''


class TestPrintAligned:
    def test_widths_fit_widest_cell(self, capsys: pytest.CaptureFixture) -> None:
        client_utils.print_aligned(['NAME', 'N'], [['a-very-long-name', '1'], ['b', '200']], right_align={1})
        header, first, second = capsys.readouterr().out.splitlines()
        assert header == 'NAME                N'
        assert first == 'a-very-long-name    1'
        assert second == 'b                 200'

    def test_indent(self, capsys: pytest.CaptureFixture) -> None:
        client_utils.print_aligned(['NAME'], [['a']], right_align=set(), indent='  ')
        assert capsys.readouterr().out.splitlines() == ['  NAME', '  a']

    def test_no_rows_prints_nothing(self, capsys: pytest.CaptureFixture) -> None:
        client_utils.print_aligned(['NAME'], [], right_align=set())
        assert capsys.readouterr().out == ''


class TestPollState:
    """poll_state() waits out a resource's pending states, tolerating transient read failures."""

    def _poll(self, responses: list[Any], monkeypatch: pytest.MonkeyPatch, timeout: float = 5) -> dict[str, Any]:
        """Run poll_state() against a canned sequence of get_request() results; an exception item is raised."""
        remaining = list(responses)

        def fake_get_request(path: str, params: dict[str, Any] | None = None) -> Any:
            resp = remaining.pop(0)
            if isinstance(resp, BaseException):
                raise resp
            return resp

        monkeypatch.setattr(hosted, 'get_request', fake_get_request)
        return hosted.poll_state('/api/db', {}, 'database', {'PENDING'}, 0, timeout, None)

    def test_returns_when_settled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = [{'database': {'state': 'PENDING'}}, {'database': {'state': 'AVAILABLE'}}]
        assert self._poll(responses, monkeypatch) == {'state': 'AVAILABLE'}

    def test_retries_failed_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = [RuntimeError('connection refused'), {'database': {'state': 'AVAILABLE'}}]
        assert self._poll(responses, monkeypatch) == {'state': 'AVAILABLE'}

    def test_daemon_exit_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(SystemExit):
            self._poll([SystemExit(1)], monkeypatch)

    def test_returns_last_read_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        responses: list[Any] = [{'database': {'state': 'PENDING'}}] * 1000
        assert self._poll(responses, monkeypatch, timeout=0.05) == {'state': 'PENDING'}


class TestDotSegments:
    """'.' and '..' are CLI path conventions, resolved before a path reaches pixeltable, where a dot is
    still the legacy component separator."""

    @pytest.mark.parametrize(
        ('path', 'expected'),
        [
            ('.', ''),
            ('..', ''),  # clamps at the root rather than escaping the catalog
            ('a/..', ''),
            ('a/b/..', 'a'),
            ('a/./b', 'a/b'),
            ('a/b/../c', 'a/c'),
            ('a/../../b', 'b'),
            ('pxt://o:d/a/..', 'pxt://o:d'),
            ('pxt://o:d/..', 'pxt://o:d'),
            ('pxt://o:d/a/../b', 'pxt://o:d/b'),
            ('a/b', 'a/b'),  # untouched when there is nothing to resolve
            ('a.b', 'a.b'),  # the legacy separator is not a navigation token
        ],
    )
    def test_resolves(self, path: str, expected: str) -> None:
        assert utils.resolve_dot_segments(path) == expected

    def test_shape_check_two_tokens(self) -> None:
        assert utils.validate_path_shape('.') is None
        assert utils.validate_path_shape('..') is None
        assert utils.validate_path_shape('a/../b') is None
        # anything else with a dot is still the legacy separator
        for bad in ('a.b', 'a/...', '...'):
            err = utils.validate_path_shape(bad)
            assert err is not None and 'separator' in err, bad


class _StubResponse:
    """Minimal HTTP response: status code, body text, and the parsed body."""

    status_code: int
    text: str

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return json.loads(self.text)


class _FlakySession:
    """Session stub whose first n_failures post() calls raise a dropped-connection error."""

    n_failures: int
    payload: dict[str, Any]
    n_calls: int

    def __init__(self, n_failures: int, payload: dict[str, Any]) -> None:
        self.n_failures = n_failures
        self.payload = payload
        self.n_calls = 0

    def post(self, url: str, *, data: str, headers: dict[str, str], timeout: int) -> _StubResponse:
        self.n_calls += 1
        if self.n_calls <= self.n_failures:
            raise requests.exceptions.ConnectionError('Connection aborted: RemoteDisconnected')
        return _StubResponse(200, self.payload)


class TestManagementClient:
    """Management API calls whose connection drops before a response is read."""

    def _install_session(
        self, monkeypatch: pytest.MonkeyPatch, n_failures: int, payload: dict[str, Any]
    ) -> _FlakySession:
        session = _FlakySession(n_failures, payload)
        monkeypatch.setattr(management_client, '_SESSION', session)
        return session

    def test_dropped_connection(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('PIXELTABLE_API_KEY', 'test-key')

        # a read operation is sent a second time, on a new connection
        session = self._install_session(monkeypatch, 1, {'database': {'db': 'main', 'state': 'AVAILABLE'}})
        get_db = GetDbRequest(org='acme', db='main')
        assert management_client.api_call(get_db) == {'database': {'db': 'main', 'state': 'AVAILABLE'}}
        assert session.n_calls == 2

        # only once, though: a second drop surfaces as an error
        session = self._install_session(monkeypatch, 2, {'database': {}})
        with pytest.raises(requests.exceptions.ConnectionError, match='RemoteDisconnected'):
            management_client.api_call(get_db)
        assert session.n_calls == 2

        # a mutating operation is never sent again: the management API may have applied it already
        session = self._install_session(monkeypatch, 1, {'database': {}})
        with pytest.raises(requests.exceptions.ConnectionError, match='RemoteDisconnected'):
            management_client.api_call(CreateDbRequest(org='acme', db='main'))
        assert session.n_calls == 1

    def test_read_ops_known(self) -> None:
        # _READ_OPS holds operation_type strings; a rename on the protocol side must not leave stale ones
        op_values = {op.value for op in ManagementOperationType}
        stale = management_client._READ_OPS - op_values
        assert len(stale) == 0, stale
