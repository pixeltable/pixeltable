import json

from ..http import get, quote_path
from ..parser import Parser, parse_cols

EPILOG = """\
Examples:
  pxt rows my_dir/my_table
  pxt rows my_dir/my_table -n 3
  pxt rows my_dir/my_table --cols id,text,score
  pxt rows my_dir/my_table -n 50 --json

Notes:
  --cols takes a comma-separated list.
  Use 'pxt describe <table>' to discover column names first.
  Unstored computed columns are skipped by default (selecting one would force
  evaluation, which may be slow or invoke external services); pass them
  explicitly via --cols to include them.
  Image cells are stringified as '<Image WxH MODE>'; other media cells return
  their stored representation (a local path for stored media, or a URL)."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt rows', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('-n', type=int, default=10, help='number of rows (default 10, max 1000)')
    ap.add_argument('--cols', help='comma-separated column subset')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    cols = parse_cols(args.cols, ap)
    cols_csv = ','.join(cols) if cols is not None else None
    resp = get(f'/api/tables/{quote_path(args.path)}/rows', params={'n': args.n, 'cols': cols_csv})

    if args.as_json:
        print(json.dumps(resp['rows'], indent=2, default=str))
        return

    columns = resp['columns']
    print('\t'.join(columns))
    for row in resp['rows']:
        print('\t'.join('' if row.get(c) is None else str(row[c]) for c in columns))
