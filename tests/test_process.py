import os
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Callable, NoReturn

import psutil
import pytest

from pixeltable.utils import process


class TestProcess:
    def test_pid_alive_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # the calling process is running, whatever the platform reports for anything else
        assert process.pid_alive(os.getpid()) is True

        # Exercise the POSIX path explicitly; on Windows pid_alive() dispatches elsewhere (see
        # test_pid_alive_dispatches_to_win32), so pin the platform to keep these cases meaningful everywhere.
        monkeypatch.setattr(process.sys, 'platform', 'linux')

        def raising(error: Exception) -> Callable[[int], NoReturn]:
            def _process(pid: int) -> NoReturn:
                raise error

            return _process

        def reporting(state: str) -> Callable[[int], SimpleNamespace]:
            return lambda pid: SimpleNamespace(status=lambda: state)

        # a pid no process holds -> gone
        monkeypatch.setattr(process.psutil, 'Process', raising(psutil.NoSuchProcess(99999)))
        assert process.pid_alive(99999) is False

        # a process whose state this one may not read (owned by another user) -> alive
        monkeypatch.setattr(process.psutil, 'Process', raising(psutil.AccessDenied(1)))
        assert process.pid_alive(1) is True

        monkeypatch.setattr(process.psutil, 'Process', reporting(psutil.STATUS_SLEEPING))
        assert process.pid_alive(4242) is True
        # 0 addresses the caller's process group and a negative value names no process, whatever psutil says
        assert process.pid_alive(0) is False
        assert process.pid_alive(-1) is False

        # a zombie has exited, so it is not alive
        monkeypatch.setattr(process.psutil, 'Process', reporting(psutil.STATUS_ZOMBIE))
        assert process.pid_alive(4242) is False

    @pytest.mark.skipif(sys.platform == 'win32', reason='Windows has no zombie processes')
    def test_pid_alive_unreaped_child(self) -> None:
        """A child that has exited but that nobody waited on is not alive."""
        proc = subprocess.Popen([sys.executable, '-c', ''], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        proc.stdout.read()  # returns at EOF, which the child reaches as it exits
        deadline = time.monotonic() + 10.0
        while process.pid_alive(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert proc.returncode is None  # nothing waited on it, so its pid is still in the process table
        os.kill(proc.pid, 0)  # raises once the pid is gone, so reaching here means the zombie is what we probe
        assert process.pid_alive(proc.pid) is False
        assert proc.wait() == 0

    def test_pid_alive_dispatches_to_win32(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # On Windows, pid_alive() must use the Win32 probe and never call os.kill(), which there maps to
        # TerminateProcess and would kill the very process being probed.
        monkeypatch.setattr(process.sys, 'platform', 'win32')
        monkeypatch.setattr(os, 'kill', lambda pid, sig: pytest.fail('os.kill() must not be called on Windows'))
        monkeypatch.setattr(process, '_win_pid_alive', lambda pid: pid == 4242)
        assert process.pid_alive(4242) is True
        assert process.pid_alive(1) is False

    @pytest.mark.skipif(sys.platform != 'win32', reason='Win32 process probe')
    def test_win_pid_alive_probe(self) -> None:
        # The current process is live; a pid that cannot be opened reads as gone.
        assert process._win_pid_alive(os.getpid()) is True
        assert process._win_pid_alive(0xFFFFFFFE) is False
