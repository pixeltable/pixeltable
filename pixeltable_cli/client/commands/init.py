"""'pxt init': mark a directory as a project root."""

import json
import pathlib
import sys
import tomllib

from ..parser import Parser

EPILOG = """\
Examples:
  pxt init                          # mark the current directory as a project root
  pxt init --json                   # the same, machine-readable
  pxt init -f                       # mark it even though a project root sits above it

Layout:
  A project root holds the application files of one project, and the modules they import:

    ~/proj/pixeltable.toml          # written by this command; marks the root
    ~/proj/ad_gen/app.py            # one application
    ~/proj/ad_gen/functions.py      # a module it imports as 'from ad_gen.functions import ...'

  Imports resolve from the root down. A file directly under the root imports its neighbors by their
  bare names; a file in a subdirectory names that subdirectory as well. Each directory from the root
  down to an application file becomes one component of its module path, so each name has to be a
  Python identifier.

Notes:
  A UDF is recorded as a module path relative to the project root, which is how a later process
  resolves it. 'pxt schema', 'pxt service' and 'pxt app' therefore require the file they are given to
  sit under a root, and say so when it does not.
  In a directory that already holds a pyproject.toml, this command adds a [tool.pixeltable] section
  to that file rather than writing a second one.
  Running this again in a directory that is already a project root reports it and changes nothing.

Exit codes:
  0  the directory is a project root
  1  error: the marker could not be written
  3  refused: a project root already sits above this directory; -f marks it anyway"""

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 3

_MARKER = 'pixeltable.toml'
_PYPROJECT = 'pyproject.toml'
_MARKER_CONTENT = """\
# Marks the root of a Pixeltable project. Application files under this directory are imported by the
# module path they have relative to it.
"""
_PYPROJECT_SECTION = """
# Marks the root of a Pixeltable project. Application files under this directory are imported by the
# module path they have relative to it.
[tool.pixeltable]
"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt init', epilog=EPILOG, usage_exit_code=EXIT_ERROR)
    ap.add_argument('-f', '--force', action='store_true', help='mark this directory even below another root')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    root = pathlib.Path.cwd()
    existing = _find_marker(root)
    if existing is not None and existing.parent == root:
        _report(root, existing, created=False, as_json=args.as_json)
        return
    if existing is not None and not args.force:
        print(
            f'pxt init: {existing.parent} is already a project root ({existing.name}), and {root} sits under '
            'it.\nA project has one root: put this directory under that one, or pass -f to make it a root of '
            'its own.',
            file=sys.stderr,
        )
        sys.exit(EXIT_REFUSED)

    marker = _write_marker(root)
    _report(root, marker, created=True, as_json=args.as_json)


def _find_marker(start: pathlib.Path) -> pathlib.Path | None:
    """Find the marker file of the nearest project root at or above start. Returns None if there is none."""
    for dir in (start, *start.parents):
        marker = dir / _MARKER
        if marker.is_file():
            return marker
        pyproject = dir / _PYPROJECT
        if pyproject.is_file() and _declares_pixeltable(pyproject):
            return pyproject
    return None


def _declares_pixeltable(pyproject: pathlib.Path) -> bool:
    """Report whether pyproject holds a [tool.pixeltable] section."""
    try:
        with open(pyproject, 'rb') as fp:
            parsed = tomllib.load(fp)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    tool = parsed.get('tool')
    return isinstance(tool, dict) and 'pixeltable' in tool


def _write_marker(root: pathlib.Path) -> pathlib.Path:
    """Mark root as a project root, and return the file that marks it."""
    pyproject = root / _PYPROJECT
    try:
        if pyproject.is_file():
            with open(pyproject, 'a', encoding='utf-8') as fp:
                fp.write(_PYPROJECT_SECTION)
            return pyproject
        marker = root / _MARKER
        marker.write_text(_MARKER_CONTENT, encoding='utf-8')
        return marker
    except OSError as e:
        print(f'pxt init: {root} could not be marked as a project root: {e}', file=sys.stderr)
        sys.exit(EXIT_ERROR)


def _report(root: pathlib.Path, marker: pathlib.Path, *, created: bool, as_json: bool) -> None:
    """Report the root, the file that marks it, and any directory whose name a module path cannot hold."""
    unusable = sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.isidentifier() and any(d.glob('*.py'))
    )
    if as_json:
        print(
            json.dumps(
                {'project_root': str(root), 'marker': str(marker), 'created': created, 'unusable_dirs': unusable},
                indent=2,
            )
        )
        return
    if created:
        print(f'project root: {root}\nwrote {marker.name}')
    else:
        print(f'project root: {root}\nalready marked by {marker.name}')
    for name in unusable:
        print(
            f"pxt init: '{name}' holds Python files, but its name is not a Python identifier, so nothing "
            'under it can be imported; rename the directory to use it in this project.',
            file=sys.stderr,
        )
