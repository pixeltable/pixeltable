import json

from ..http import get, quote_path
from ..parser import Parser

EPILOG = """\
Examples:
  pxt history my_dir/my_table
  pxt history my_dir/my_table -n 5      # last 5 versions only
  pxt history my_dir/my_table --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt history', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('-n', type=int, default=None, help='show at most N most recent versions')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = get(f'/api/tables/{quote_path(args.path)}/history', params={'n': args.n})
    versions = resp['versions']

    if args.as_json:
        print(json.dumps(versions, indent=2))
        return

    print('version\tcreated_at\tchange_type\tinserts\tupdates\tdeletes\terrors\tschema_change')
    for v in versions:
        print(
            '\t'.join(
                str(v.get(k) if v.get(k) is not None else '')
                for k in (
                    'version',
                    'created_at',
                    'change_type',
                    'inserts',
                    'updates',
                    'deletes',
                    'errors',
                    'schema_change',
                )
            )
        )
