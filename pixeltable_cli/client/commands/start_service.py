from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt service start', description='Start a service in a cloud-hosted database.')
    parser.add_argument('service_uri', help='Service URI: pxt://org:db/services/<name>')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from ..cloud import parse_service_uri, poll_svc, print_service
    from ..http import post

    try:
        org_slug, db_slug, svc_name = parse_service_uri(args.service_uri, prog='pxt service start')
        resp = post(f'/api/cloud/orgs/{org_slug}/dbs/{db_slug}/services/{svc_name}/start', {})
        svc = resp.get('service', resp) if isinstance(resp, dict) else {}
        if svc.get('state') in ('STARTING', 'DEPLOYING'):
            svc = poll_svc(org_slug, db_slug, svc_name, frozenset({'STARTING', 'DEPLOYING'}), f"Service '{svc_name}' is starting...")
        if args.json_output:
            print(json.dumps(svc))
        else:
            print_service(svc)
    except SystemExit:
        raise
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)
