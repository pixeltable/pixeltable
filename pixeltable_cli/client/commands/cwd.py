from ..parser import Parser
from ..utils import display_path, post_request, validate_path_arg

EPILOG = """\
Examples:
  pxt cwd pxt://org:db/my_dir      # absolute: a hosted directory
  pxt cwd /my_dir                  # absolute: from the local catalog root
  pxt cwd my_dir                   # relative: my_dir under the current working directory
  pxt cwd                          # clear the working directory

Notes:
  The working directory is prepended to relative paths in subsequent commands; a pxt:// URI or a
  leading '/' is absolute and ignores it.
  It is scoped to the invoking terminal (keyed by the shell's session), so separate terminals don't
  share it and it does not leak into subprocesses or agents you launch -- those run under their own
  session with no working directory. Scripts and agents should use absolute paths and ignore it."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt cwd', epilog=EPILOG)
    ap.add_argument('uri', nargs='?', default=None, help='directory to set as the working directory; omit to clear')
    args = ap.parse_args(argv)

    # omitting the argument clears the working directory; root ('/') is equivalent to no working directory
    uri = args.uri if args.uri is not None else '/'
    resp = post_request('/api/cwd', {'uri': validate_path_arg(uri)})
    if resp['uri'] is None:
        print('working directory cleared')
    else:
        print(f'working directory: {display_path(resp["uri"])}')
