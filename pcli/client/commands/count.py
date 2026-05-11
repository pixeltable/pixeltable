import json

from ..http import post
from ..parser import Parser

EPILOG = """\
Examples:
  pcli count my_dir.my_table
  pcli count my_dir/my_table --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli count', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = post('/pcli/v0/count', {'path': args.path})

    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(resp['count'])
