import os
import sys

import pytest

from pixeltable.service import proxy_daemon


class TestProxyDaemon:
    def test_pid_alive_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Exercise the POSIX os.kill() path explicitly; on Windows _pid_alive() dispatches elsewhere (see
        # test_pid_alive_dispatches_to_win32), so pin the platform to keep these os.kill cases meaningful everywhere.
        monkeypatch.setattr(proxy_daemon.sys, 'platform', 'linux')

        # ProcessLookupError (no such pid) -> gone
        monkeypatch.setattr(os, 'kill', lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))
        assert proxy_daemon._pid_alive(99999) is False

        # PermissionError (pid exists, owned by another user) -> alive
        monkeypatch.setattr(os, 'kill', lambda pid, sig: (_ for _ in ()).throw(PermissionError()))
        assert proxy_daemon._pid_alive(1) is True

        # A non-lookup OSError is treated as gone
        monkeypatch.setattr(
            os, 'kill', lambda pid, sig: (_ for _ in ()).throw(OSError(22, 'The parameter is incorrect'))
        )
        assert proxy_daemon._pid_alive(0) is False

        # SystemError is treated as gone
        monkeypatch.setattr(os, 'kill', lambda pid, sig: (_ for _ in ()).throw(SystemError()))
        assert proxy_daemon._pid_alive(0) is False

    def test_pid_alive_dispatches_to_win32(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # On Windows, _pid_alive() must use the Win32 probe and never call os.kill(), which there maps to
        # TerminateProcess and would kill the very process being probed.
        monkeypatch.setattr(proxy_daemon.sys, 'platform', 'win32')
        monkeypatch.setattr(os, 'kill', lambda pid, sig: pytest.fail('os.kill() must not be called on Windows'))
        monkeypatch.setattr(proxy_daemon, '_win_pid_alive', lambda pid: pid == 4242)
        assert proxy_daemon._pid_alive(4242) is True
        assert proxy_daemon._pid_alive(1) is False

    @pytest.mark.skipif(sys.platform != 'win32', reason='Win32 process probe')
    def test_win_pid_alive_probe(self) -> None:
        # The current process is live; a pid that cannot be opened reads as gone.
        assert proxy_daemon._win_pid_alive(os.getpid()) is True
        assert proxy_daemon._win_pid_alive(0xFFFFFFFE) is False
