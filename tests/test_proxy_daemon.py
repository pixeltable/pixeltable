import os

import pytest

from pixeltable.service import proxy_daemon


class TestProxyDaemon:
    def test_pid_alive_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ProcessLookupError (POSIX, no such pid) -> gone
        monkeypatch.setattr(os, 'kill', lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))
        assert proxy_daemon._pid_alive(99999) is False

        # PermissionError (pid exists, owned by another user) -> alive
        monkeypatch.setattr(os, 'kill', lambda pid, sig: (_ for _ in ()).throw(PermissionError()))
        assert proxy_daemon._pid_alive(1) is True

        # On Windows os.kill(pid, 0) raises OSError (WinError 87), not ProcessLookupError, for a missing pid
        monkeypatch.setattr(
            os, 'kill', lambda pid, sig: (_ for _ in ()).throw(OSError(22, 'The parameter is incorrect'))
        )
        assert proxy_daemon._pid_alive(0) is False

        # SystemError can surface on Windows for an unknown pid -> gone
        monkeypatch.setattr(os, 'kill', lambda pid, sig: (_ for _ in ()).throw(SystemError()))
        assert proxy_daemon._pid_alive(0) is False
