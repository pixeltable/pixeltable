import sys


def confirm_or_exit(prompt: str, force: bool) -> None:
    """Prompt for yes/no on stdin; refuse non-tty unless --force."""
    if force:
        return
    if not sys.stdin.isatty():
        print(f'pcli: refusing to proceed without --force/-f (no TTY for confirmation): {prompt}', file=sys.stderr)
        sys.exit(2)
    sys.stderr.write(f'{prompt} [y/N] ')
    sys.stderr.flush()
    ans = sys.stdin.readline().strip().lower()
    if ans not in ('y', 'yes'):
        print('aborted', file=sys.stderr)
        sys.exit(1)
