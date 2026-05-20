import json

from ..http import get
from ..parser import Parser

EPILOG = """\
Examples:
  pxt idxs                         # every index across every table
  pxt idxs my_dir/my_table         # indexes on one table
  pxt idxs --embedding             # only embedding indexes (skip auto-created btrees)
  pxt idxs --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt idxs', epilog=EPILOG)
    ap.add_argument('path', nargs='?', default=None, help='if omitted: every index in the catalog')
    ap.add_argument('--embedding', action='store_true', dest='embedding_only', help='restrict to embedding indexes')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = get('/api/indexes', params={'path': args.path or None, 'embedding': args.embedding_only or None})
    entries = resp['entries']

    if args.as_json:
        print(json.dumps(entries, indent=2))
        return

    for e in entries:
        cols = ','.join(e['columns'])
        extra = ''
        if e['index_type'] == 'embedding':
            metric = e['metric'] if e.get('metric') is not None else ''
            embedding = e['embedding'] if e.get('embedding') is not None else ''
            extra = f'\t{metric}\t{embedding}'
        print(f'{e["table"]}\t{e["name"]}\t{e["index_type"]}\t{cols}{extra}')
