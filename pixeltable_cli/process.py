"""Finding, probing and stopping processes by pid. Stdlib plus psutil, so the pxt client imports it cheaply."""

import contextlib
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

    For validating a pid that arrives as data, such as one decoded from a json file. bool is an int, and 0 and
    negative values address the caller's process group rather than one process, so passing them to os.kill()
    signals more than the intended target.
    """
    return type(value) is int and value > 0


def process_timestamp(pid: int) -> float | None:
    """The creation time of pid, or None if it cannot be read."""
    if pid <= 0:
        return None
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OSError):
        return None


def pid_alive(pid: int) -> bool:
    """True if pid is a live process. An already-exited but unreaped child (zombie) counts as dead."""
    # 0 and negatives name a process group rather than a process, and psutil rejects a negative outright
    if pid <= 0:
        return False
    if sys.platform == 'win32':
        return _win_pid_alive(pid)
    try:
        # a zombie has terminated, so it must not read as running; psutil reports that state wherever
        # zombies exist, whereas /proc/<pid>/stat is Linux-only
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except (psutil.Error, OSError):
        return True  # the process exists, but this one cannot read its state


def find_process(pid: int, started_at: float | None) -> psutil.Process | None:
    """Return the live process with pid and creation time started_at, or None.

    A pid alone does not identify a process: the OS reuses the pid of an exited process. A record written without a
    creation time has started_at None, which matches any live process with pid.
    """
    if not pid_alive(pid):
        return None
    try:
        proc = psutil.Process(pid)
        return proc if started_at is None or proc.create_time() == started_at else None
    except (psutil.Error, OSError):
        return None


def terminate_and_wait(proc: psutil.Process, timeout: float) -> None:
    """Terminate proc, wait up to timeout seconds for it to exit, then kill it.

    psutil never signals a process that reused proc.pid after the creation of proc, and wait() reaps proc if it is a
    child of this process.
    """
    try:
        proc.terminate()
        proc.wait(timeout)
    except psutil.TimeoutExpired:
        with contextlib.suppress(psutil.Error, OSError):
            proc.kill()
    except (psutil.Error, OSError):
        # gone or exiting: on Linux, psutil re-raises FileNotFoundError when /proc/<pid>/stat is gone before
        # /proc/<pid>, as during an exit
        pass
