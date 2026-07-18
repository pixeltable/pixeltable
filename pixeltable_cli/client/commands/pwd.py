from ..parser import Parser
from ..utils import display_path, get_request

EPILOG = """\
Examples:
  pxt pwd                          # print the working directory, or '(no working directory)' when unset

Notes:
  Set or clear the working directory with 'pxt cwd'."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt pwd', epilog=EPILOG)
    ap.parse_args(argv)
    resp = get_request('/api/cwd')
    print(display_path(resp['uri']) if resp['uri'] is not None else '(no working directory)')
