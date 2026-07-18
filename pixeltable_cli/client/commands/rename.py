import json

from ...utils import validate_path_shape
from ..parser import Parser
from ..utils import display_path, post_request

EPILOG = """\
Examples:
  pxt rename my_dir/old_name new_name
  pxt rename my_dir/old_name new_name -n         # dry-run

Notes:
  Paths are slash-separated. Renames in place: parent directory is preserved.
  Use 'pxt mv' to move across directories.
  <new_name> is just the new leaf name, not a full path."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt rename', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('new_name', help='new leaf name (no slashes or dots)')
    ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    err = validate_path_shape(args.path)
    if err is not None:
        ap.error(err)
    if '/' in args.new_name or '.' in args.new_name:
        ap.error(f'new_name must be a leaf name (no "/" or "."); got {args.new_name!r}')

    parent = args.path.rsplit('/', 1)[0] if '/' in args.path else ''
    dst = f'{parent}/{args.new_name}' if parent else args.new_name

    if args.dry_run:
        print(f'would rename {args.path} -> {dst}')
        return

    resp = post_request('/api/move', {'path': args.path, 'new_path': dst})
    if args.as_json:
        print(json.dumps(resp, indent=2))
    else:
        print(f'renamed {display_path(resp["path"])} -> {display_path(resp["new_path"])}')
