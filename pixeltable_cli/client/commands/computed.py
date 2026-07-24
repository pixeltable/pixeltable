import json

from ...utils import validate_path_shape
from ..parser import Parser
from ..utils import display_path, get_request

EPILOG = """\
Examples:
  pxt computed                          # every computed column across every table
  pxt computed my_dir/my_table          # computed columns of one table
  pxt computed my_dir                   # every table under a directory, recursively
  pxt computed pxt://org:db             # every table in a hosted database
  pxt computed pxt://org:db/my_dir      # every table under a directory, recursively
  pxt computed --json

This is equivalent to 'pxt columns --computed [path]' but easier to remember."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt computed', epilog=EPILOG)
    ap.add_argument(
        'path',
        nargs='?',
        default=None,
        help='a table, or a directory to walk recursively; if omitted, every table in the catalog',
    )
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)
    if args.path is not None:
        err = validate_path_shape(args.path)
        if err is not None:
            ap.error(err)

    resp = get_request('/api/columns', params={'path': args.path or None, 'computed': True})
    entries = resp['entries']

    if args.as_json:
        print(json.dumps(entries, indent=2))
        return

    for e in entries:
        expr = e['computed_with'] if e.get('computed_with') is not None else ''
        print(f'{display_path(e["table"])}\t{e["column"]}\t{e["type_"]}\t{expr}')
