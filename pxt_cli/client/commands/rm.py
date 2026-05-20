import json

from ..confirm import confirm_or_exit
from ..http import post, quote_path
from ..parser import Parser

EPILOG = """\
Examples:
  pxt rm my_dir -f                # remove an empty directory
  pxt rm my_dir -r -f             # recursive: also remove contained tables/views/subdirs
  pxt rm my_dir -n -r             # dry-run

Notes:
  Refuses tables/views; use 'pxt drop' for those.
  Without -r, fails if the directory is non-empty.
  Without -f, confirmation is read from the terminal; non-interactive callers must pass -f."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt rm', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('-r', '--recursive', action='store_true')
    ap.add_argument('-f', '--force', action='store_true', help='skip confirmation')
    ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    if args.dry_run:
        suffix = ' (recursive: contained tables/subdirs also removed)' if args.recursive else ''
        print(f'would remove {args.path} (if it is a directory){suffix}')
        return

    confirm_or_exit(f'remove directory {args.path}?', args.force)

    resp = post(f'/api/dirs/{quote_path(args.path)}/drop', {'cascade': args.recursive})
    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(f'removed {resp["path"]}')
