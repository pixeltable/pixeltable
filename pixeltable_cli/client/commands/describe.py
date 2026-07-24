import json

from ..parser import Parser
from ..utils import get_request, validate_path_arg

EPILOG = """\
Examples:
  pxt describe my_dir/my_table
  pxt describe my_dir/my_table --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt describe', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = get_request('/api/tables/describe', params={'path': validate_path_arg(args.path)})
    print(json.dumps(resp['metadata'], indent=2) if args.as_json else resp['text'])
