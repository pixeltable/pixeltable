import json

from ..http import post
from ..parser import Parser

EPILOG = """\
Examples:
  pcli ls
  pcli ls some_dir
  pcli ls --tree
  pcli ls -l some_dir
  pcli ls --counts                 # include row counts (runs queries)
  pcli ls some_dir --json"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli ls', epilog=EPILOG)
    ap.add_argument('path', nargs='?', default='')
    ap.add_argument('--tree', action='store_true')
    ap.add_argument('-l', '--long', action='store_true')
    ap.add_argument('--counts', action='store_true', help='include row counts (runs queries)')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = post('/pcli/v0/ls', {
        'path': args.path, 'tree': args.tree, 'long': args.long, 'counts': args.counts,
    })

    if args.as_json:
        print(json.dumps(resp, indent=2))
        return

    if args.tree:
        _print_tree(resp['tree'])
        return

    for e in resp['entries']:
        cols = [e['path'], e['kind']]
        if args.counts and e.get('num_rows') is not None:
            cols.append(str(e['num_rows']))
        if args.long:
            cols.append(str(e.get('num_cols') or ''))
            cols.append(str(e.get('last_version') or ''))
            cols.append(e.get('flags', '') or '-')
        print('\t'.join(cols))


def _print_tree(node: dict, prefix: str = '') -> None:
    entries = node.get('entries', [])
    for i, child in enumerate(entries):
        last = i == len(entries) - 1
        bar = '└── ' if last else '├── '
        print(f"{prefix}{bar}{child['name']}\t{child['kind']}")
        if child['kind'] == 'directory':
            _print_tree({'entries': child.get('entries', [])}, prefix + ('    ' if last else '│   '))
