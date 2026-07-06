import json

from ..http import get, quote_path
from ..parser import Parser, parse_cols

EPILOG = """\
Examples:
  pxt get my_dir/my_table 42                    # single-column PK, int value
  pxt get my_dir/my_table some_string_id        # single-column PK, string value
  pxt get my_dir/my_table 42 abc                # composite PK (2 cols), in declared PK order
  pxt get my_dir/my_table 42 --cols id,text     # restrict to listed columns
  pxt get my_dir/my_table 42 --json

Notes:
  PK values are coerced to int or float when they parse as numbers; otherwise they stay
  as strings. There is no way to force a string PK that looks like a number; if your PK
  column is typed as string but the value is '42', the server will reject the type mismatch.
  Use 'pxt describe <table>' to see the primary_key columns and their order.
  Unstored computed columns are skipped by default; pass them explicitly via --cols to
  include them.
  The table must have a primary key declared."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt get', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('pk', nargs='*', help='primary key values in PK column order')
    ap.add_argument('--cols', help='comma-separated column subset')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    if not args.pk:
        ap.error('pk values are required for table row lookup')
    # Reject empty/whitespace-only PK tokens: argparse accepts pxt get t '' (or a stray
    # space), and an empty PK would silently produce a 'no row found' that masks the typo.
    if any(v.strip() == '' for v in args.pk):
        ap.error('PK values must not be empty or whitespace-only')
    cols = parse_cols(args.cols, ap)
    cols_csv = ','.join(cols) if cols is not None else None
    # PK coercion (numeric strings -> int/float) happens on the server side; the URL only
    # carries strings.
    resp = get(f'/api/tables/{quote_path(args.path)}/row', params={'pk': args.pk, 'cols': cols_csv})

    if args.as_json:
        print(json.dumps(resp, indent=2, default=str))
        return

    if resp['row'] is None:
        pk_pairs = ', '.join(f'{n}={v!r}' for n, v in zip(resp['pk_columns'], args.pk))
        print(f'no row found for {{{pk_pairs}}}')
        return
    for k, v in resp['row'].items():
        print(f'{k}\t{v}')
