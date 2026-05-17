import json

from ..http import post
from ..parser import Parser

EPILOG = """\
Examples:
  pcli rename my_dir/old_name new_name
  pcli rename my_dir/old_name new_name -n         # dry-run

Notes:
  Paths are slash-separated. Renames in place: parent directory is preserved.
  Use 'pcli mv' to move across directories.
  <new_name> is just the new leaf name, not a full path."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli rename', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('new_name', help='new leaf name (no slashes or dots)')
    ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    if '/' in args.new_name or '.' in args.new_name:
        ap.error(f'new_name must be a leaf name (no "/" or "."); got {args.new_name!r}')

    parent = args.path.rsplit('/', 1)[0] if '/' in args.path else ''
    dst = f'{parent}/{args.new_name}' if parent else args.new_name

    if args.dry_run:
        print(f'would rename {args.path} -> {dst}')
        return

    resp = post('/pcli/v0/move', {'path': args.path, 'new_path': dst})
    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(f'renamed {resp["path"]} -> {resp["new_path"]}')
