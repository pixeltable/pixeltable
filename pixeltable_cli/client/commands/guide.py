"""`pxt guide` - print an agent guide to this version of Pixeltable, or list the guides."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

from ..parser import Parser

EPILOG = """\
Examples:
  pxt guide              # list the guides, one line each
  pxt guide tables       # print the guide to tables
  pxt guide --json       # the guides as {area: summary}
  pxt guide --path       # the directory of the guide files, to read or search them

A guide states how this version of Pixeltable behaves in one area: what its APIs do, their defaults, errors
and known issues."""


def pixeltable_dir() -> pathlib.Path:
    # find_spec() locates the package without importing it, which would slow down the command
    spec = importlib.util.find_spec('pixeltable')
    assert spec is not None and spec.submodule_search_locations is not None
    return pathlib.Path(next(iter(spec.submodule_search_locations)))


def guides_dir() -> pathlib.Path:
    return pixeltable_dir() / 'agent_guides'


def guide_paths() -> dict[str, pathlib.Path]:
    return {path.stem: path for path in sorted(guides_dir().glob('*.md')) if path.name != 'README.md'}


def run(argv: list[str]) -> None:
    paths = guide_paths()
    parser = Parser(
        prog='pxt guide', description='print an agent guide to this version of Pixeltable, or list them', epilog=EPILOG
    )
    parser.add_argument('area', nargs='?', choices=list(paths), help='the guide to print; omit it to list the guides')
    parser.add_argument(
        '--path',
        action='store_true',
        help="print the path of the guide's file, or of the guides' directory, instead of the text",
    )
    parser.add_argument('--json', action='store_true', dest='as_json', help='Emit JSON output')
    args = parser.parse_args(argv)
    if args.path:
        path = paths[args.area] if args.area is not None else guides_dir()
        print(json.dumps({'path': str(path)}, indent=2) if args.as_json else path)
        return
    if args.area is not None:
        text = paths[args.area].read_text(encoding='utf-8')
        if args.as_json:
            print(json.dumps({'area': args.area, 'text': text}, indent=2))
        else:
            sys.stdout.write(text)
        return
    if args.as_json:
        print(json.dumps({area: _summary(path) for area, path in paths.items()}, indent=2))
        return
    width = max(len(area) for area in paths)
    for area, path in paths.items():
        print(f'{area:<{width}}  {_summary(path)}')


def _summary(path: pathlib.Path) -> str:
    """Return the first nonblank line after the guide's title."""
    lines = path.read_text(encoding='utf-8').splitlines()
    return next(line for line in lines[1:] if line.strip() != '')
