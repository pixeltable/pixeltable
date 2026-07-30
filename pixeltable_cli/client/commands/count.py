import json

from ..parser import Parser
from ..utils import get_request, validate_path_arg

EPILOG = """\
Examples:
  pxt count my_dir/my_table
  pxt count my_dir/my_table --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt count', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = get_request('/api/tables/count', params={'path': validate_path_arg(args.path)})

    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(resp['count'])
