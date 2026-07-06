from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator

if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl


class FileLock:
    """An advisory lock shared across processes via a lock file, implemented with OS-specific primitives:
    - Posix: shared/exclusive flock
    - Windows: msvcrt provides only exclusive byte-range locks

    The lock is associated with the wrapped file descriptor, not with the calling thread.

    Not thread-safe.
    """

    _fd: int

    def __init__(self, path: Path) -> None:
        # O_RDWR is required for msvcrt.locking on Windows; the file's contents are never read or written.
        self._fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)

    def shared(self) -> contextlib.AbstractContextManager[None]:
        """Hold a shared lock for the duration of the block."""
        return self._hold(exclusive=False)

    def exclusive(self) -> contextlib.AbstractContextManager[None]:
        """Hold an exclusive lock for the duration of the block."""
        return self._hold(exclusive=True)

    @contextlib.contextmanager
    def _hold(self, *, exclusive: bool) -> Iterator[None]:
        self._acquire(exclusive=exclusive)
        try:
            yield
        finally:
            self._release()

    def _acquire(self, *, exclusive: bool) -> None:
        if sys.platform == 'win32':
            # msvcrt has no shared mode, so a shared request is taken exclusively. LK_LOCK blocks (retrying for
            # ~10s before raising OSError).
            os.lseek(self._fd, 0, os.SEEK_SET)
            msvcrt.locking(self._fd, msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(self._fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

    def _release(self) -> None:
        if sys.platform == 'win32':
            os.lseek(self._fd, 0, os.SEEK_SET)
            msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def close(self) -> None:
        os.close(self._fd)
