import sys
from collections.abc import Callable


def stdin_is_a_tty() -> bool:
    """Like `sys.stdin.isatty()`, but on Windows distinguishes real consoles from NUL/other
    character devices. msvcrt's `isatty` returns nonzero for any char device, so `subprocess.DEVNULL`
    (which maps to NUL) is misreported as a TTY; GetConsoleMode() succeeds only on real consoles.
    """
    if not sys.stdin.isatty():
        return False
    if sys.platform != 'win32':
        return True
    import ctypes
    from ctypes import wintypes

    handle = ctypes.windll.msvcrt._get_osfhandle(sys.stdin.fileno())
    mode = wintypes.DWORD()
    return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))


def confirm_or_exit(
    prompt: str, force: bool, *, refused_exit_code: int = 2, on_refusal: Callable[[], None] | None = None
) -> None:
    """Prompt for yes/no on stdin; refuse non-tty unless --force. Both refusals exit with refused_exit_code.

    on_refusal runs just before the non-tty refusal exits, for a caller that reports the refusal itself.
    """
    if force:
        return
    if not stdin_is_a_tty():
        if on_refusal is not None:
            on_refusal()
        print(f'pxt: refusing to proceed without --force/-f (no TTY for confirmation): {prompt}', file=sys.stderr)
        sys.exit(refused_exit_code)
    sys.stderr.write(f'{prompt} [y/N] ')
    sys.stderr.flush()
    ans = sys.stdin.readline().strip().lower()
    if ans not in ('y', 'yes'):
        print('aborted', file=sys.stderr)
        sys.exit(refused_exit_code)
