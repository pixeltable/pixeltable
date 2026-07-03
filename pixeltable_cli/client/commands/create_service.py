from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt service create', description='Create a service from a table in a cloud-hosted database.')
    parser.add_argument('table_uri', help='Table URI: pxt://org:db/tables/<path>')
    parser.add_argument('--name', required=True, help='Service name')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers (default: 1)')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable.catalog.path import Path as PxtPath
    from pixeltable.share.deploy_client import service_create

    try:
        p = PxtPath.parse(args.table_uri)
        if p.org is None or p.db is None:
            parser.error('table_uri must be pxt://org:db/tables/<path>')
        # components: ('tables', 'my_table') → table_path = 'my_table'
        # or plain ('my_table',) if user omitted the 'tables/' prefix
        comps = list(p.components)
        if comps and comps[0] == 'tables':
            comps = comps[1:]
        table_path = '.'.join(comps)
        if not table_path:
            parser.error('table_uri must include a table path, e.g. pxt://org:db/tables/my_table')
        service_create(
            p.org,
            p.db,
            args.name,
            table_path,
            workers_min=args.workers,
            workers_max=args.workers,
            json_output=args.json_output,
        )
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)
