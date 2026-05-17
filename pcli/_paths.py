"""Stdlib-only helpers (imported by `pcli.probe`, which must not pull in pxt or pydantic)."""

import os


def _resolved_home() -> str | None:
    home = os.environ.get('PIXELTABLE_HOME') or os.path.expanduser('~/.pixeltable')
    try:
        return os.path.realpath(home)
    except OSError:
        return None


def redact_home(path: str | None) -> str | None:
    """Replace the resolved PIXELTABLE_HOME prefix in `path` with the literal `$PIXELTABLE_HOME`.

    Paths outside PIXELTABLE_HOME are returned unchanged. Symlinks on both sides are resolved
    so eg `/Users/me/.pixeltable/...` and `/private/Users/me/.pixeltable/...` match on macOS.
    """
    if path is None:
        return None
    home_resolved = _resolved_home()
    if home_resolved is None:
        return path
    try:
        target = os.path.realpath(path)
    except OSError:
        return path
    if target == home_resolved:
        return '$PIXELTABLE_HOME'
    if target.startswith(home_resolved + os.sep):
        return '$PIXELTABLE_HOME' + target[len(home_resolved) :]
    return path


def redact_home_in_text(text: str) -> str:
    """String-substitute every occurrence of the resolved PIXELTABLE_HOME prefix in arbitrary text.

    Unlike redact_home(), which expects `text` to be a single path, this is for free-form
    output (log tails, exception messages, multi-line error bodies) where the home path
    may appear anywhere. No symlink resolution on the embedded paths: log lines typically
    contain the resolved form already.
    """
    home_resolved = _resolved_home()
    if home_resolved is None or home_resolved == '':
        return text
    return text.replace(home_resolved, '$PIXELTABLE_HOME')
