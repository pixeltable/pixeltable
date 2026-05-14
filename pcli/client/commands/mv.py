import json

from ..http import post
from ..parser import Parser

EPILOG = """\
Examples:
  pcli mv my_dir.my_table other_dir              # move to /other_dir/my_table
  pcli mv my_dir/my_table /                      # move to root
  pcli mv my_dir.my_table other_dir -n           # dry-run

Notes:
  The leaf name is preserved; only the parent directory changes. Use 'pcli rename' to change the leaf name.
  <new_dir> can be '' or '/' for the root directory."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli mv', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('new_dir', help='destination directory (use "" or "/" for root)')
    ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    src = args.path.replace('.', '/')
    leaf = src.rsplit('/', 1)[-1]
    new_dir = args.new_dir.strip('/')
    dst = f'{new_dir}/{leaf}' if new_dir else leaf

    if args.dry_run:
        print(f'would move {args.path} -> {dst}')
        return

    resp = post('/pcli/v0/move', {'path': args.path, 'new_path': dst})
    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(f'moved {resp["path"]} -> {resp["new_path"]}')
