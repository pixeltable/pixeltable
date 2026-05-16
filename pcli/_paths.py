"""Stdlib-only helpers (imported by `pcli.probe`, which must not pull in pxt or pydantic)."""

import os


def redact_home(path: str | None) -> str | None:
    """Replace the resolved PIXELTABLE_HOME prefix in `path` with the literal `$PIXELTABLE_HOME`.

    Paths outside PIXELTABLE_HOME are returned unchanged. Symlinks on both sides are resolved
    so eg `/Users/me/.pixeltable/...` and `/private/Users/me/.pixeltable/...` match on macOS.
    """
    if path is None:
        return None
    home = os.environ.get('PIXELTABLE_HOME') or os.path.expanduser('~/.pixeltable')
    try:
        home_resolved = os.path.realpath(home)
        target = os.path.realpath(path)
    except OSError:
        return path
    if target == home_resolved:
        return '$PIXELTABLE_HOME'
    if target.startswith(home_resolved + os.sep):
        return '$PIXELTABLE_HOME' + target[len(home_resolved) :]
    return path
