import json

from ..http import get, quote_path
from ..parser import Parser

EPILOG = """\
Examples:
  pxt count my_dir/my_table
  pxt count my_dir/my_table --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt count', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = get(f'/api/tables/{quote_path(args.path)}/count')

    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(resp['count'])
