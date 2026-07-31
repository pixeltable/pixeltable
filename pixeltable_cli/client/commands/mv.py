import json

from ...utils import validate_path_shape
from ..parser import Parser
from ..utils import display_path, post_request

EPILOG = """\
Examples:
  pxt mv my_dir/my_table other_dir              # move to other_dir/my_table
  pxt mv my_dir/my_table /                      # move to the catalog root
  pxt mv my_dir/my_table .                      # move to the working directory
  pxt mv my_dir/my_table other_dir -n           # dry-run

Notes:
  Paths are slash-separated. The name is preserved; only the parent directory changes.
  Use 'pxt rename' to change the name.
  A leading '/' makes <new_dir> absolute; without one it is taken under the working directory.
  Use '/' on its own for the catalog root."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt mv', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('new_dir', help='destination directory (use "/" for the catalog root)')
    ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    err = validate_path_shape(args.path)
    if err is not None:
        ap.error(err)
    err = validate_path_shape(args.new_dir)
    if err is not None:
        ap.error(err)
    name = args.path.rsplit('/', 1)[-1]
    parent = args.new_dir.rstrip('/')
    if parent != '':
        dst = f'{parent}/{name}'
    elif args.new_dir == '':
        dst = name  # the working directory
    else:
        dst = f'/{name}'  # the catalog root

    resp = post_request('/api/move', {'path': args.path, 'new_path': dst, 'dry_run': args.dry_run})
    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        verb = 'would move' if args.dry_run else 'moved'
        print(f'{verb} {display_path(resp["path"])} -> {display_path(resp["new_path"])}')
