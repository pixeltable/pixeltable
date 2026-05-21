from __future__ import annotations

import sys
import webbrowser

from pxt_cli.client.utils import base_url, ensure_running

from ..parser import Parser

EPILOG = """\
Examples:
  pxt dashboard               # ensure the daemon is running, print and open the URL
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt dashboard', description='print and open the dashboard URL', epilog=EPILOG)
    parser.parse_args(argv)
    try:
        ensure_running()
    except RuntimeError as e:
        print(f'pxt: {e}', file=sys.stderr)
        sys.exit(1)
    url = base_url()
    print(url)
    webbrowser.open(url)
