from __future__ import annotations

import json
import sys

import pixeltable as pxt
from pixeltable.serving import deploy

from ..parser import Parser

EPILOG = """\
To deploy a configured deployment:
  pxt deploy <deployment-name>
"""


def run(argv: list[str]) -> None:
    parser = Parser(
        prog='pxt deploy',
        description='Deploy the service in the specified deployment configuration to Pixeltable cloud.',
        epilog=EPILOG,
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
