import json

from ..http import post
from ..parser import Parser

EPILOG = """\
Examples:
  pcli computed                          # every computed column across every table
  pcli computed my_dir.my_table          # computed columns of one table
  pcli computed --json

This is equivalent to 'pcli columns --computed [path]' but easier to remember."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli computed', epilog=EPILOG)
    ap.add_argument('path', nargs='?', default=None, help='if omitted: every table in the catalog')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = post('/pcli/v0/columns', {'path': args.path, 'computed_only': True})
    entries = resp['entries']

    if args.as_json:
        print(json.dumps(entries, indent=2))
        return

    for e in entries:
        expr = e.get('computed_with') or ''
        print(f"{e['table']}\t{e['column']}\t{e['type_']}\t{expr}")
