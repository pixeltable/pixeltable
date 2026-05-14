import json

from ..http import get
from ..parser import Parser

EPILOG = """\
Examples:
  pcli status
  pcli status --sizes               # also report media/file_cache disk usage (slower)
  pcli status --json"""


def _fmt_size(n: int | None) -> str:
    if n is None:
        return '-'
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f'{n:.1f}{unit}' if unit != 'B' else f'{n}B'
        n /= 1024
    return f'{n:.1f}PB'


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli status', epilog=EPILOG)
    ap.add_argument('--sizes', action='store_true', help='include media/file_cache directory sizes')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    s = get('/pcli/v0/status' + ('?sizes=1' if args.sizes else ''))

    if args.as_json:
        print(json.dumps(s, indent=2))
        return

    print(f"pxt_version     {s['pxt_version']}")
    print(f"daemon_pid      {s['pid']}")
    print(f"daemon_started  {s['started_at']}")
    print(f"home            {s['home'] or '-'}")
    print(f"db_url          {s['db_url'] or '-'}")
    print(f"media_dir       {s['media_dir'] or '-'}  ({_fmt_size(s['media_size_bytes'])})")
    print(f"file_cache_dir  {s['file_cache_dir'] or '-'}  ({_fmt_size(s['file_cache_size_bytes'])})")
    print(f"total_tables    {s['total_tables']}")
    print(f"total_errors    {s['total_errors']}")
