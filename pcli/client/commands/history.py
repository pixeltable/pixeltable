import json

from ..http import post
from ..parser import Parser

EPILOG = """\
Examples:
  pcli history my_dir/my_table
  pcli history my_dir/my_table -n 5      # last 5 versions only
  pcli history my_dir/my_table --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli history', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('-n', type=int, default=None, help='show at most N most recent versions')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = post('/pcli/v0/history', {'path': args.path, 'n': args.n})
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
