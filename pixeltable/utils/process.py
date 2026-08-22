"""Liveness of a process we started, as far as its pid can tell us."""

import sys

import psutil


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


def is_pid(value: object) -> bool:
    """True if value can name a process: an int above 0.

    bool is an int, and 0 and negative values address the caller's process group rather than one process, so
    passing them to os.kill() signals more than the intended target.
    """
    return type(value) is int and value > 0


def process_timestamp(pid: int) -> float | None:
    """The creation time of pid, or None if it cannot be read."""
    if not is_pid(pid):
        return None
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OSError):
        return None


def pid_alive(pid: int) -> bool:
    """True if pid is a live process. An already-exited but unreaped child (zombie) counts as dead."""
    if sys.platform == 'win32':
        return _win_pid_alive(pid)
    if not is_pid(pid):
        return False
    try:
        # a zombie has terminated, so it must not read as running; psutil reports that state wherever
        # zombies exist, whereas /proc/<pid>/stat is Linux-only
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except (psutil.Error, OSError):
        return True  # the process exists, but this one cannot read its state
