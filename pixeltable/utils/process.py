"""Liveness of a process we started, as far as its pid can tell us."""

import os
import sys


def _win_pid_alive(pid: int) -> bool:
    """Windows liveness check via the Win32 API.

    os.kill(pid, 0) cannot probe liveness on Windows: CPython maps os.kill() to
    OpenProcess(PROCESS_ALL_ACCESS) + TerminateProcess(handle, sig), so signal 0 would terminate a live
    process, and OpenProcess raises Access-denied (WinError 5) for an already-exited process, which would
    read as alive. Instead, open the process with only SYNCHRONIZE rights and check whether its handle is
    signaled: a running process's handle is unsignaled (WAIT_TIMEOUT); an exited one is signaled.
    """
    # ctypes.wintypes exists only on Windows, so import it inside this platform-guarded path.
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)  # type: ignore[attr-defined]  # Windows-only
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    synchronize = 0x00100000  # SYNCHRONIZE access right, the minimum needed to wait on the process handle
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return False  # no such process
    try:
        # WaitForSingleObject returns WAIT_TIMEOUT (0x102) while the process runs; it returns WAIT_OBJECT_0
        # (0) once the process has exited and its handle becomes signaled.
        return kernel32.WaitForSingleObject(handle, 0) == 0x102
    finally:
        kernel32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    """True if pid is a live process. An already-exited but unreaped child (zombie) counts as dead."""
    if sys.platform == 'win32':
        return _win_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except (OSError, SystemError):
        return False
    # os.kill(pid, 0) also succeeds for a zombie (exited but not yet reaped). A zombie has terminated, so
    # treat it as dead; otherwise a process that we launched and that has already exited reads as running.
    try:
        with open(f'/proc/{pid}/stat', encoding='ascii') as f:
            # the state is the field after the parenthesized comm, which may itself contain spaces/parens
            state = f.read().rsplit(') ', 1)[1].split()[0]
    except (OSError, IndexError):
        return True  # no /proc (non-Linux) or a transient read race: trust the os.kill result
    return state != 'Z'
