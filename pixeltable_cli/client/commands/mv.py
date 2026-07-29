import json

from ...utils import validate_path_shape
from ..parser import Parser
from ..utils import display_path, post_request

EPILOG = """\
Examples:
  pxt mv my_dir/my_table other_dir              # move to other_dir/my_table
  pxt mv my_dir/my_table /                      # move to the catalog root
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
    name = args.path.rsplit('/', 1)[-1]
    # a leading '/' makes the destination absolute
    is_absolute = args.new_dir.startswith('/')
    new_dir = args.new_dir.strip('/')
    if new_dir != '':
        err = validate_path_shape(new_dir)
        if err is not None:
            ap.error(err)
    dst = f'{new_dir}/{name}' if new_dir != '' else name
    if is_absolute:
        dst = f'/{dst}'

    if args.dry_run:
        print(f'would move {args.path} -> {dst}')
        return

    resp = post_request('/api/move', {'path': args.path, 'new_path': dst})
    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(f'moved {display_path(resp["path"])} -> {display_path(resp["new_path"])}')
