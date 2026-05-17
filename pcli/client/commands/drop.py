import json

from ..confirm import confirm_or_exit
from ..http import post
from ..parser import Parser

EPILOG = """\
Examples:
  pcli drop my_dir/my_table -f                  # drop a table (no confirmation)
  pcli drop my_dir/my_table -n                  # dry-run: print what would happen, don't act
  pcli drop my_dir/my_table --cascade -f        # also drop dependent views
  pcli drop my_dir/my_table --json -f

Notes:
  Refuses directories; use 'pcli rm' for those.
  Without --cascade, will fail if dependent views exist (the error names them).
  Without -f, confirmation is read from the terminal; non-interactive callers must pass -f."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli drop', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('-f', '--force', action='store_true', help='skip confirmation')
    ap.add_argument('--cascade', action='store_true', help='also drop dependent views')
    ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    if args.dry_run:
        suffix = ' (cascade: dependent views also dropped)' if args.cascade else ''
        print(f'would drop table {args.path}{suffix}')
        return

    confirm_or_exit(f'drop table {args.path}?', args.force)

    resp = post('/pcli/v0/drop', {'path': args.path, 'cascade': args.cascade, 'is_dir': False})
    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(f'dropped {resp["path"]}')
