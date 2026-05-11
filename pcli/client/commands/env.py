import json

from ..http import get
from ..parser import Parser

EPILOG = """\
Examples:
  pcli env
  pcli env --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli env', epilog=EPILOG)
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = get('/pcli/v0/env')

    if args.as_json:
        print(json.dumps(resp, indent=2))
        return

    print(f"config_file  {resp['config_file'] or '-'}")
    for k in sorted(resp['env_vars']):
        print(f'{k}={resp["env_vars"][k]}')
