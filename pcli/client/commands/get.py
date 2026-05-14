import json

from ..http import post
from ..parser import Parser


def _coerce(s: str) -> object:
    """Try int, then float, then JSON literal; fall back to the raw string."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


EPILOG = """\
Examples:
  pcli get my_dir.my_table 42                    # single-column PK, int value
  pcli get my_dir.my_table some_string_id        # single-column PK, string value
  pcli get my_dir.my_table 42 abc                # composite PK (2 cols), in declared PK order
  pcli get my_dir.my_table 42 --json

Notes:
  PK values are auto-coerced: int -> float -> JSON literal -> string.
  To force a string that looks like a number, pass a JSON literal: pcli get t '"42"'
  Use 'pcli describe <table>' to see the table's primary_key columns and their order.
  The table must have a primary key declared."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli get', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('pk', nargs='+', help='primary key values in PK column order')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    pk_values = [_coerce(v) for v in args.pk]
    resp = post('/pcli/v0/get', {'path': args.path, 'pk': pk_values})

    if args.as_json:
        print(json.dumps(resp, indent=2, default=str))
        return

    if resp['row'] is None:
        pk_pairs = ', '.join(f'{n}={v!r}' for n, v in zip(resp['pk_columns'], pk_values))
        print(f'no row found for {{{pk_pairs}}}')
        return
    for k, v in resp['row'].items():
        print(f'{k}\t{v}')
