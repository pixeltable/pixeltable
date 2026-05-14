import json

from ..http import post
from ..parser import Parser

EPILOG = """\
Examples:
  pcli idxs                         # every index across every table
  pcli idxs my_dir.my_table         # indexes on one table
  pcli idxs --embedding             # only embedding indexes (skip auto-created btrees)
  pcli idxs --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli idxs', epilog=EPILOG)
    ap.add_argument('path', nargs='?', default=None, help='if omitted: every index in the catalog')
    ap.add_argument('--embedding', action='store_true', dest='embedding_only', help='restrict to embedding indexes')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = post('/pcli/v0/idxs', {'path': args.path, 'embedding_only': args.embedding_only})
    entries = resp['entries']

    if args.as_json:
        print(json.dumps(entries, indent=2))
        return

    for e in entries:
        cols = ','.join(e['columns'])
        extra = ''
        if e['index_type'] == 'embedding':
            extra = f'\t{e.get("metric") or ""}\t{e.get("embedding") or ""}'
        print(f'{e["table"]}\t{e["name"]}\t{e["index_type"]}\t{cols}{extra}')
