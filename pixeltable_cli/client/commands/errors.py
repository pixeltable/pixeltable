import json

from ..parser import Parser
from ..utils import get_request, validate_path_arg

EPILOG = """\
Examples:
  pxt errors my_dir/my_table
  pxt errors my_dir/my_table --col embedding
  pxt errors my_dir/my_table --json

Note: the table must have a primary key (declared via primary_key= and pxt.Required[...])."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt errors', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('--col', help='filter to a single computed column')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = get_request('/api/tables/errors', params={'path': validate_path_arg(args.path), 'col': args.col})

    if args.as_json:
        print(json.dumps(resp['entries'], indent=2, default=str))
        return

    for e in resp['entries']:
        pk = '{' + ', '.join(f'{k}: {v!r}' for k, v in e['pk'].items()) + '}'
        errmsg = e['errormsg'] if e['errormsg'] is not None else ''
        print(f'{pk}\t{e["column"]}\t{e["errortype"]}\t{errmsg}')
