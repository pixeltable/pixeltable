"""`pxt guide` - print an agent guide to this version of Pixeltable, or list the guides."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

from ..parser import Parser

EPILOG = """\
Examples:
  pxt guide              # list the guides, one line each
  pxt guide tables       # print the guide to tables

A guide states how this version of Pixeltable behaves in one area: what its APIs do, their defaults, errors
and known issues. 'pxt skills install' also copies the guides into the skill's guides/ directory."""


def pixeltable_dir() -> pathlib.Path:
    # find_spec() locates the package without importing it, which would slow down the command
    spec = importlib.util.find_spec('pixeltable')
    assert spec is not None and spec.submodule_search_locations is not None
    return pathlib.Path(next(iter(spec.submodule_search_locations)))


def guide_paths() -> dict[str, pathlib.Path]:
    guides_dir = pixeltable_dir() / 'agent_guides'
    return {path.stem: path for path in sorted(guides_dir.glob('*.md')) if path.name != 'README.md'}


def run(argv: list[str]) -> None:
    paths = guide_paths()
    parser = Parser(
        prog='pxt guide', description='print an agent guide to this version of Pixeltable, or list them', epilog=EPILOG
    )
    parser.add_argument('area', nargs='?', choices=list(paths), help='the guide to print; omit it to list the guides')
    args = parser.parse_args(argv)
    if args.area is not None:
        sys.stdout.write(paths[args.area].read_text(encoding='utf-8'))
        return
    width = max(len(area) for area in paths)
    for area, path in paths.items():
        print(f'{area:<{width}}  {_summary(path)}')


def _summary(path: pathlib.Path) -> str:
    """Return the first nonblank line after the guide's title."""
    lines = path.read_text(encoding='utf-8').splitlines()
    return next(line for line in lines[1:] if line.strip() != '')
