import json

from ..http import post
from ..parser import Parser, parse_cols


def _coerce(s: str) -> object:
    """Coerce numeric-looking PK tokens to int or float; everything else stays a string."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


EPILOG = """\
Examples:
  pcli get my_dir/my_table 42                    # single-column PK, int value
  pcli get my_dir/my_table some_string_id        # single-column PK, string value
  pcli get my_dir/my_table 42 abc                # composite PK (2 cols), in declared PK order
  pcli get my_dir/my_table 42 --cols id,text     # restrict to listed columns
  pcli get my_dir/my_table 42 --json

Notes:
  PK values are coerced to int or float when they parse as numbers; otherwise they stay
  as strings. There is no way to force a string PK that looks like a number; if your PK
  column is typed as string but the value is '42', the server will reject the type mismatch.
  Use 'pcli describe <table>' to see the primary_key columns and their order.
  Unstored computed columns are skipped by default; pass them explicitly via --cols to
  include them.
  The table must have a primary key declared."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli get', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('pk', nargs='+', help='primary key values in PK column order')
    ap.add_argument('--cols', help='comma-separated column subset')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    # Reject empty/whitespace-only PK tokens: argparse accepts pcli get t '' (or a stray
    # space), and an empty PK would silently produce a 'no row found' that masks the typo.
    if any(v.strip() == '' for v in args.pk):
        ap.error('PK values must not be empty or whitespace-only')
    pk_values = [_coerce(v) for v in args.pk]
    cols = parse_cols(args.cols, ap)
    resp = post('/pcli/v0/get', {'path': args.path, 'pk': pk_values, 'cols': cols})

    if args.as_json:
        print(json.dumps(resp, indent=2, default=str))
        return

    if resp['row'] is None:
        pk_pairs = ', '.join(f'{n}={v!r}' for n, v in zip(resp['pk_columns'], pk_values))
        print(f'no row found for {{{pk_pairs}}}')
        return
    for k, v in resp['row'].items():
        print(f'{k}\t{v}')
