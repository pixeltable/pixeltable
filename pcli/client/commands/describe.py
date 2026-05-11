import json

from ..http import post
from ..parser import Parser

EPILOG = """\
Examples:
  pcli describe my_dir.my_table
  pcli describe my_dir/my_table --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli describe', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = post('/pcli/v0/describe', {'path': args.path})
    print(json.dumps(resp['metadata'], indent=2) if args.as_json else resp['text'])
