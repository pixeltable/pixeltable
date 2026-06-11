import json

from ...utils import validate_path_shape
from ..http import get
from ..parser import Parser

EPILOG = """\
Examples:
  pxt columns                          # every column across every table
  pxt columns my_dir/my_table          # columns of one table
  pxt columns --computed               # only computed columns, all tables
  pxt columns my_dir/my_table --computed --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt columns', epilog=EPILOG)
    ap.add_argument('path', nargs='?', default=None, help='if omitted: every table in the catalog')
    ap.add_argument('--computed', action='store_true', dest='computed_only')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)
    if args.path is not None:
        err = validate_path_shape(args.path)
        if err is not None:
            ap.error(err)

    resp = get('/api/columns', params={'path': args.path or None, 'computed': args.computed_only or None})
    entries = resp['entries']

    if args.as_json:
        print(json.dumps(entries, indent=2))
        return

    for e in entries:
        flag = 'computed' if e['is_computed'] else 'stored'
        expr = e['computed_with'] if e.get('computed_with') is not None else ''
        print(f'{e["table"]}\t{e["column"]}\t{e["type_"]}\t{flag}\t{expr}')
