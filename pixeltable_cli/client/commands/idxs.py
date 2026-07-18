import json

from ...utils import validate_path_shape
from ..parser import Parser
from ..utils import display_path, get_request

EPILOG = """\
Examples:
  pxt idxs                         # every index across every table
  pxt idxs my_dir/my_table         # indexes on one table
  pxt idxs my_dir                  # every table under a directory, recursively
  pxt idxs pxt://org:db            # every table in a hosted database
  pxt idxs pxt://org:db/my_dir     # every table under a directory, recursively
  pxt idxs --embedding             # only embedding indexes (skip auto-created btrees)
  pxt idxs --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt idxs', epilog=EPILOG)
    ap.add_argument(
        'path',
        nargs='?',
        default=None,
        help='a table, or a directory to walk recursively; if omitted, every table in the catalog',
    )
    ap.add_argument('--embedding', action='store_true', dest='embedding_only', help='restrict to embedding indexes')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)
    if args.path is not None:
        err = validate_path_shape(args.path)
        if err is not None:
            ap.error(err)

    resp = get_request('/api/indexes', params={'path': args.path or None, 'embedding': args.embedding_only or None})
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
        print(f'{display_path(e["table"])}\t{e["name"]}\t{e["index_type"]}\t{cols}{extra}')
