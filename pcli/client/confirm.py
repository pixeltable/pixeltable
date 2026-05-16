import sys


def _stdin_is_real_tty() -> bool:
    """Like `sys.stdin.isatty()`, but on Windows distinguishes real consoles from NUL/other
    character devices. msvcrt's `isatty` returns nonzero for any char device, so `subprocess.DEVNULL`
    (which maps to NUL) is misreported as a TTY — `GetConsoleMode` succeeds only on real consoles.
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


def confirm_or_exit(prompt: str, force: bool) -> None:
    """Prompt for yes/no on stdin; refuse non-tty unless --force."""
    if force:
        return
    if not _stdin_is_real_tty():
        print(f'pcli: refusing to proceed without --force/-f (no TTY for confirmation): {prompt}', file=sys.stderr)
        sys.exit(2)
    sys.stderr.write(f'{prompt} [y/N] ')
    sys.stderr.flush()
    ans = sys.stdin.readline().strip().lower()
    if ans not in ('y', 'yes'):
        print('aborted', file=sys.stderr)
        sys.exit(1)
