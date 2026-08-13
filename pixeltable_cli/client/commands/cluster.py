"""`pxt cluster {create,list,status,update,delete}` - manage dedicated db clusters (paid orgs)."""

from __future__ import annotations

import argparse
import json

from ..hosted import exit_unless_reached, parse_org_uri, poll_cluster, print_cluster
from ..parser import Parser
from ..utils import get_request, post_request

EPILOG = """\
Examples:
  pxt cluster create pxt://org mycluster --size PS_10
  pxt cluster list pxt://org
  pxt cluster status pxt://org mycluster
  pxt cluster update pxt://org mycluster --size PS_40
  pxt cluster delete pxt://org mycluster
  pxt cluster delete pxt://org mycluster --force
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt cluster', description='manage dedicated db clusters', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)

    p = sub.add_parser('create', help='create a dedicated db cluster')
    p.add_argument('org_uri', help='Org URI: pxt://org')
    p.add_argument('name', help='Cluster name, unique within the org')
    p.add_argument('--size', required=True, help="PlanetScale cluster size, e.g. 'PS_10'")
    p.add_argument('--region', default=None, help='Region (default: us-east)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('list', help='list db clusters for an org')
    p.add_argument('org_uri', help='Org URI: pxt://org')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('status', help='show status of a db cluster')
    p.add_argument('org_uri', help='Org URI: pxt://org')
    p.add_argument('name', help='Cluster name')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('update', help='resize a db cluster')
    p.add_argument('org_uri', help='Org URI: pxt://org')
    p.add_argument('name', help='Cluster name')
    p.add_argument('--size', required=True, help="target PlanetScale cluster size, e.g. 'PS_40'")
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('delete', help='delete a db cluster')
    p.add_argument('org_uri', help='Org URI: pxt://org')
    p.add_argument('name', help='Cluster name')
    p.add_argument('--force', action='store_true', help='also delete every database on the cluster')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    args = parser.parse_args(argv)

    if args.action == 'create':
        _create(args)
    elif args.action == 'list':
        _list(args)
    elif args.action == 'status':
        _status(args)
    elif args.action == 'update':
        _update(args)
    elif args.action == 'delete':
        _delete(args)


def _create(args: argparse.Namespace) -> None:
    org = parse_org_uri(args.org_uri, prog='pxt cluster create')
    body = {'org': org, 'cluster': args.name, 'size': args.size}
    if args.region is not None:
        body['region'] = args.region
    resp = post_request('/api/clusters', body)
    result = resp.get('cluster', resp) if isinstance(resp, dict) else {}
    if result.get('state') == 'PROVISIONING':
        result = poll_cluster(org, args.name, {'PROVISIONING'}, f"Cluster '{args.name}' is provisioning...")
    if args.json_output:
        print(json.dumps(result))
    else:
        print_cluster(result)
    exit_unless_reached(result, 'ACTIVE', f'creating cluster {args.name!r}')


def _list(args: argparse.Namespace) -> None:
    org = parse_org_uri(args.org_uri, prog='pxt cluster list')
    resp = get_request('/api/clusters', {'org': org})
    clusters = resp.get('clusters', []) if isinstance(resp, dict) else []
    if args.json_output:
        print(json.dumps(clusters))
    elif not clusters:
        print('No clusters.')
    else:
        for cluster in clusters:
            print_cluster(cluster)


def _status(args: argparse.Namespace) -> None:
    org = parse_org_uri(args.org_uri, prog='pxt cluster status')
    resp = get_request('/api/cluster', {'org': org, 'cluster': args.name})
    result = resp.get('cluster', resp) if isinstance(resp, dict) else {}
    if args.json_output:
        print(json.dumps(result))
    else:
        print_cluster(result)


def _update(args: argparse.Namespace) -> None:
    org = parse_org_uri(args.org_uri, prog='pxt cluster update')
    resp = post_request('/api/cluster/update', {'org': org, 'cluster': args.name, 'size': args.size})
    result = resp.get('cluster', resp) if isinstance(resp, dict) else {}
    if result.get('state') == 'RESIZING':
        result = poll_cluster(org, args.name, {'RESIZING'}, f"Cluster '{args.name}' is resizing...")
    if args.json_output:
        print(json.dumps(result))
    else:
        print_cluster(result)
    exit_unless_reached(result, 'ACTIVE', f'resizing cluster {args.name!r}')


def _delete(args: argparse.Namespace) -> None:
    org = parse_org_uri(args.org_uri, prog='pxt cluster delete')
    post_request('/api/cluster/delete', {'org': org, 'cluster': args.name, 'force': args.force})
    if args.json_output:
        print(json.dumps({'deleted': args.name}))
    else:
        print(f"Deleted cluster '{args.name}'.")
