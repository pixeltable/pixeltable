import json

from ..http import post
from ..parser import Parser

EPILOG = """\
Examples:
  pcli columns                          # every column across every table
  pcli columns my_dir/my_table          # columns of one table
  pcli columns --computed               # only computed columns, all tables
  pcli columns my_dir/my_table --computed --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli columns', epilog=EPILOG)
    ap.add_argument('path', nargs='?', default=None, help='if omitted: every table in the catalog')
    ap.add_argument('--computed', action='store_true', dest='computed_only')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = post('/pcli/v0/columns', {'path': args.path, 'computed_only': args.computed_only})
    entries = resp['entries']

    if args.as_json:
        print(json.dumps(entries, indent=2))
        return

    for e in entries:
        flag = 'computed' if e['is_computed'] else 'stored'
        expr = e['computed_with'] if e.get('computed_with') is not None else ''
        print(f'{e["table"]}\t{e["column"]}\t{e["type_"]}\t{flag}\t{expr}')
