import json

from ..confirm import confirm_or_exit
from ..http import post, quote_path
from ..parser import Parser

EPILOG = """\
Examples:
  pxt revert my_dir/my_table -f                 # undo the last op
  pxt revert my_dir/my_table --steps 3 -f       # roll back 3 versions
  pxt revert my_dir/my_table -n                 # dry-run: print the intended revert (no execution)

Notes:
  Each revert undoes exactly one op (data + schema). --steps repeats the call N times.
  Reverts both data and schema together; pxt's versioning model doesn't separate them.
  This operation is irreversible. Use 'pxt history <table>' first to see what would be undone.
  Without -f, confirmation is read from the terminal; non-interactive callers must pass -f."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt revert', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('--steps', type=int, default=1, help='number of consecutive reverts (default 1)')
    ap.add_argument('-f', '--force', action='store_true', help='skip confirmation')
    ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    if args.steps < 1:
        ap.error('--steps must be >= 1')

    if args.dry_run:
        print(f'would revert {args.path} by {args.steps} version(s)')
        return

    confirm_or_exit(f'revert {args.path} by {args.steps} version(s)? this is irreversible.', args.force)

    resp = post(f'/api/tables/{quote_path(args.path)}/revert', {'steps': args.steps})
    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(f'reverted {resp["path"]}: v{resp["from_version"]} -> v{resp["to_version"]}')
