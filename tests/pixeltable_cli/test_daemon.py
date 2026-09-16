"""Tests for how a cli client adopts, replaces or spawns its daemon.

ensure_running() checks every responder on the port, whichever path produced it: one already answering,
one that answers only within the startup grace window, and one that came up after a spawn. The tests
monkeypatch client_utils, so no daemon runs here.
"""

import pathlib

import pytest

from pixeltable_cli.client import utils as client_utils

# Real identity values are too tied to the host environment to assert against, so tests pin this dict
# via _patch_identity() and pass responses built from it, overriding a field to provoke a mismatch.
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


class TestResponderChecks:
    def test_spawned_responder_serving_another_project_replaced(self) -> None:
        """A spawned daemon defers to whoever already holds the port, so the responder is checked too:
        one serving another project is replaced rather than adopted."""
        with pytest.MonkeyPatch.context() as m:
            _patch_identity(m, {})
            m.setattr(client_utils, 'project_root', lambda: '/project')
            responders = iter(
                [_health_payload(pid=100, project_root='/other'), _health_payload(pid=200, project_root='/project')]
            )
            m.setattr(client_utils, 'fetch_health', lambda *a, **kw: None)
            m.setattr(client_utils, 'read_pidfile', lambda: None)
            actions: list[tuple[str, int] | str] = []
            m.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: actions.append(('kill', pid)))
            m.setattr(client_utils, 'spawn_detached', lambda: actions.append('spawn'))
            m.setattr(client_utils, 'wait_for_health', lambda timeout=15.0: next(responders))

            client_utils.ensure_running()
            assert actions == ['spawn', ('kill', 100), 'spawn']

    def test_slow_daemon_serving_another_project_replaced(self) -> None:
        """A daemon that answers only within the grace window is checked like any other: one serving
        another project is replaced rather than adopted."""
        with pytest.MonkeyPatch.context() as m:
            _patch_identity(m, {})
            m.setattr(client_utils, 'project_root', lambda: '/project')
            slow = _health_payload(pid=100, project_root='/other')
            replacement = _health_payload(pid=200, project_root='/project')
            m.setattr(client_utils, 'fetch_health', lambda *a, **kw: None)
            m.setattr(client_utils, 'read_pidfile', lambda: 100)
            m.setattr(client_utils, '_pid_alive', lambda pid: True)
            m.setattr(client_utils, '_pid_is_our_daemon', lambda pid: True)
            m.setattr(client_utils, '_await_health', lambda timeout: slow)
            actions: list[tuple[str, int] | str] = []
            m.setattr(client_utils, 'kill_and_wait', lambda pid, timeout=5.0: actions.append(('kill', pid)))
            m.setattr(client_utils, 'spawn_detached', lambda: actions.append('spawn'))
            m.setattr(client_utils, 'wait_for_health', lambda timeout=15.0: replacement)

            client_utils.ensure_running()
            assert actions == [('kill', 100), 'spawn']


class TestPidHygiene:
    """A pid that names no process must never be signaled."""

    # 0 and negative values name a process group rather than a process (-1 every process we may signal),
    # and bool is an int in Python
    @pytest.mark.parametrize('reported_pid', ['not-a-pid', 0, -1, True])
    def test_identity_mismatch_invalid_pid_refuses(self, monkeypatch: pytest.MonkeyPatch, reported_pid: object) -> None:
        """Identity drift but the responder reports a pid no process can have: refuse to restart (no kill,
        no spawn) rather than act on an untrustworthy pid."""
        _patch_identity(monkeypatch, {'pxt_version': 'NEW'})
        health = _health_payload(pxt_version='OLD')
        health['pid'] = reported_pid
        monkeypatch.setattr(client_utils, 'fetch_health', lambda *a, **kw: health)
        monkeypatch.setattr(client_utils, 'read_pidfile', lambda: 100)
        monkeypatch.setattr(
            client_utils, 'kill_and_wait', lambda pid, timeout=5.0: pytest.fail('must not kill an invalid pid')
        )
        monkeypatch.setattr(client_utils, 'spawn_detached', lambda: pytest.fail('must not spawn'))

        with pytest.raises(RuntimeError, match='invalid pid'):
            client_utils.ensure_running()

    @pytest.mark.parametrize('content', ['not-an-int', '0', '-1'])
    def test_pidfile_malformed(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, content: str) -> None:
        """A pidfile holding anything but a real process id reads as absent, so nothing signals on it."""
        monkeypatch.setattr(client_utils, 'pidfile_path', lambda: str(tmp_path / 'bogus.pid'))
        with open(client_utils.pidfile_path(), 'w', encoding='utf-8') as f:
            f.write(content)
        assert client_utils.read_pidfile() is None
