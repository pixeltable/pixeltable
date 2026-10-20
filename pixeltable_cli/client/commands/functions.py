"""`pxt functions` - list Pixeltable's functions, or a module's, with their signatures."""

from __future__ import annotations

import json
import os
import sys
import typing

import pixeltable as pxt
from pixeltable.types import FunctionKind

from ..parser import Parser
from ..utils import EXIT_ERROR

TYPE_NAMES = (
    'Array',
    'Audio',
    'Binary',
    'Bool',
    'Date',
    'Document',
    'Float',
    'Image',
    'Int',
    'Json',
    'String',
    'Timestamp',
    'UUID',
    'Video',
)

EPILOG = """\
Examples:
  pxt functions                     # every function, aggregate and iterator in pixeltable.functions
  pxt functions video               # those in pixeltable.functions.video
  pxt functions --on Json           # the methods and properties of Json expressions
  pxt functions --search detect     # those whose path or docstring contains 'detect'
  pxt functions app --kind udf      # the UDFs of app.py in this directory
  pxt functions --on Image --json   # the same records, machine-readable

Each line shows how to call a function, then the first paragraph of its docstring."""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt functions', description='list functions with their signatures', epilog=EPILOG)
    parser.add_argument(
        'module', nargs='?', help="a submodule of pixeltable.functions, such as 'video', or a module path"
    )
    parser.add_argument('--on', choices=TYPE_NAMES, metavar='TYPE', help='list the methods and properties of TYPE')
    parser.add_argument('--search', metavar='TEXT', help='list the functions whose path or docstring contains TEXT')
    parser.add_argument('--kind', choices=typing.get_args(FunctionKind), help='list one kind only')
    parser.add_argument('--json', action='store_true', dest='as_json', help='Emit JSON output')
    args = parser.parse_args(argv)

    # import modules in the working directory, as python -c 'import app' does
    sys.path.insert(0, os.getcwd())
    try:
        functions = pxt.list_functions(
            args.module, on=None if args.on is None else getattr(pxt, args.on), search=args.search, kind=args.kind
        )
    except pxt.Error as e:
        print(f'pxt functions: {e}', file=sys.stderr)
        sys.exit(EXIT_ERROR)

    if args.as_json:
        print(json.dumps(functions, indent=2))
    elif len(functions) == 0:
        print('pxt functions: no function matches', file=sys.stderr)
    else:
        print(functions)
