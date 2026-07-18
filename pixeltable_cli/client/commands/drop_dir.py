import json

from ..confirm import confirm_or_exit
from ..parser import Parser
from ..utils import display_path, post_request, validate_path_arg

EPILOG = """\
Examples:
  pxt drop-dir my_dir -f                # remove an empty directory
  pxt drop-dir my_dir -r -f             # recursive: also remove contained tables/views/subdirs
  pxt drop-dir my_dir -n -r             # dry-run

Notes:
  Refuses tables/views; use 'pxt drop' for those.
  Without -r, fails if the directory is non-empty.
  Without -f, confirmation is read from the terminal; non-interactive callers must pass -f."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt drop-dir', epilog=EPILOG)
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

    resp = post_request('/api/dirs/drop', {'path': validate_path_arg(args.path), 'cascade': args.recursive})
    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(f'removed {display_path(resp["path"])}')
