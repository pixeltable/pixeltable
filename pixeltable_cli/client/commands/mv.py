import json

from ...utils import validate_path_shape
from ..parser import Parser
from ..utils import display_path, post_request

EPILOG = """\
Examples:
  pxt mv my_dir/my_table other_dir              # move to other_dir/my_table
  pxt mv my_dir/my_table /                      # move to root
  pxt mv my_dir/my_table other_dir -n           # dry-run

Notes:
  Paths are slash-separated. The leaf name is preserved; only the parent directory changes.
  Use 'pxt rename' to change the leaf name.
  <new_dir> can be '' or '/' for the root directory."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt mv', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('new_dir', help='destination directory (use "" or "/" for root)')
    ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    err = validate_path_shape(args.path)
    if err is not None:
        ap.error(err)
    leaf = args.path.rsplit('/', 1)[-1]
    # '' and '/' both mean root; otherwise validate after stripping outer slashes.
    new_dir = args.new_dir.strip('/')
    if new_dir != '':
        err = validate_path_shape(new_dir)
        if err is not None:
            ap.error(err)
    dst = f'{new_dir}/{leaf}' if new_dir else leaf

    if args.dry_run:
        print(f'would move {args.path} -> {dst}')
        return

    resp = post_request('/api/move', {'path': args.path, 'new_path': dst})
    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(f'moved {display_path(resp["path"])} -> {display_path(resp["new_path"])}')
