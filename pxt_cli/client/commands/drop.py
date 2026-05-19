import json

from ..confirm import confirm_or_exit
from ..http import post, quote_path
from ..parser import Parser

EPILOG = """\
Examples:
  pxt drop my_dir/my_table -f                  # drop a table (no confirmation)
  pxt drop my_dir/my_table -n                  # dry-run: print what would happen, don't act
  pxt drop my_dir/my_table --cascade -f        # also drop dependent views
  pxt drop my_dir/my_table --json -f

Notes:
  Refuses directories; use 'pxt rm' for those.
  Without --cascade, will fail if dependent views exist (the error names them).
  Without -f, confirmation is read from the terminal; non-interactive callers must pass -f."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt drop', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('-f', '--force', action='store_true', help='skip confirmation')
    ap.add_argument('--cascade', action='store_true', help='also drop dependent views')
    ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    if args.dry_run:
        # Don't claim 'table' - we haven't validated the path's kind. The server's drop_table
        # call is the one that enforces it, and we're skipping the round-trip in dry-run mode.
        suffix = ' (cascade: dependent views also dropped)' if args.cascade else ''
        print(f'would drop {args.path} (if it is a table or view){suffix}')
        return

    confirm_or_exit(f'drop table {args.path}?', args.force)

    resp = post(f'/api/tables/{quote_path(args.path)}/drop', {'cascade': args.cascade})
    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(f'dropped {resp["path"]}')
