from __future__ import annotations

import argparse
import json
import sys

import pixeltable as pxt
from pixeltable.serving import deploy

EPILOG = """\
To deploy a configured deployment:
  pxt deploy <deployment-name>
"""


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that appends the epilog to stderr on error."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        sys.stderr.write(f'\npxt: error: {message}\n')
        if self.epilog is not None:
            sys.stderr.write(f'\n{self.epilog}\n')
        sys.exit(2)


def run(argv: list[str]) -> None:
    parser = _Parser(
        prog='pxt deploy',
        description='Deploy the service in the specified deployment configuration to Pixeltable cloud.',
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('deployment', help='Name of the target deployment')
    parser.add_argument('--json', action='store_true', dest='json', help='Emit machine-readable JSON output')
    args = parser.parse_args(argv)

    try:
        deploy.build_deploy_bundle(args.deployment)
    except pxt.Error as e:
        if args.json:
            print(json.dumps({'status': 'error', 'message': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)
