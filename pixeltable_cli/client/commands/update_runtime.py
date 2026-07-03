from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt update-runtime', description='Trigger a runtime rebuild for a cloud-hosted database.')
    parser.add_argument('db_uri', help='Database URI: pxt://org:db or just the db slug')
    parser.add_argument('--image', dest='runtime_image', default=None, help='Pin a specific runtime image')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable.share.deploy_client import database_update_runtime

    try:
        uri = args.db_uri
        if uri.startswith('pxt://'):
            from pixeltable.catalog.path import Path as PxtPath

            p = PxtPath.parse(uri, allow_empty_path=True)
            if p.db is None:
                parser.error('db_uri must be pxt://org:db')
            database_update_runtime(
                p.db, org_slug=p.org, runtime_image=args.runtime_image, json_output=args.json_output
            )
        else:
            database_update_runtime(uri, runtime_image=args.runtime_image, json_output=args.json_output)
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)
