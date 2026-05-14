import json

from ..http import post
from ..parser import Parser

EPILOG = """\
Examples:
  pcli errors my_dir.my_table
  pcli errors my_dir.my_table --col embedding
  pcli errors my_dir.my_table --json

Note: the table must have a primary key (declared via primary_key= and pxt.Required[...])."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli errors', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('--col', help='filter to a single computed column')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = post('/pcli/v0/errors', {'path': args.path, 'col': args.col})

    if args.as_json:
        print(json.dumps(resp['entries'], indent=2, default=str))
        return

    for e in resp['entries']:
        pk = '{' + ', '.join(f'{k}: {v!r}' for k, v in e['pk'].items()) + '}'
        print(f'{pk}\t{e["column"]}\t{e["errortype"]}\t{e["errormsg"] or ""}')
