import json

from ..http import post
from ..parser import Parser

EPILOG = """\
Columns under -l:
  rows      number of rows (only with --counts)
  cols      number of currently-visible user columns
  version   last committed version number
  flags     'c' = has at least one computed column
            'i' = has at least one index

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

    resp = post('/pcli/v0/ls', {'path': args.path, 'tree': args.tree, 'counts': args.counts})

    if args.as_json:
        print(json.dumps(resp, indent=2))
        return

    if args.tree:
        _print_tree(resp['tree'])
        return

    headers = ['path', 'kind']
    right_align = set()
    if args.counts:
        headers.append('rows')
        right_align.add(len(headers) - 1)
    if args.long:
        headers.extend(['cols', 'version', 'flags'])
        right_align.update({len(headers) - 3, len(headers) - 2})

    rows: list[list[str]] = []
    for e in resp['entries']:
        row = [e['path'], e['kind']]
        if args.counts:
            row.append('' if e.get('num_rows') is None else str(e['num_rows']))
        if args.long:
            row.append('' if e.get('num_cols') is None else str(e['num_cols']))
            row.append('' if e.get('last_version') is None else str(e['last_version']))
            row.append(e.get('flags') or '-')
        rows.append(row)

    _print_aligned(headers, rows, right_align)


def _print_aligned(headers: list[str], rows: list[list[str]], right_align: set[int]) -> None:
    if not rows:
        return
    widths = [max(len(c) for c in col) for col in zip(headers, *rows)]

    def fmt(r: list[str]) -> str:
        cells = [c.rjust(w) if i in right_align else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths))]
        return '  '.join(cells).rstrip()

    print(fmt(headers))
    for r in rows:
        print(fmt(r))


def _print_tree(node: dict, prefix: str = '') -> None:
    entries = node.get('entries', [])
    for i, child in enumerate(entries):
        last = i == len(entries) - 1
        bar = '└── ' if last else '├── '
        print(f'{prefix}{bar}{child["name"]}  {child["kind"]}')
        if child['kind'] == 'directory':
            _print_tree({'entries': child.get('entries', [])}, prefix + ('    ' if last else '│   '))
