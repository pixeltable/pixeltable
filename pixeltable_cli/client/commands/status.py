import json

from ..http import get
from ..parser import Parser

EPILOG = """\
Examples:
  pxt status
  pxt status --sizes               # also report media/file_cache disk usage (slower)
  pxt status --json"""


def _fmt_size(n: int | None) -> str:
    if n is None:
        return '-'
    val: float = n
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if val < 1024:
            return f'{val:.1f}{unit}' if unit != 'B' else f'{int(val)}B'
        val /= 1024
    return f'{val:.1f}PB'


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt status', epilog=EPILOG)
    ap.add_argument('--sizes', action='store_true', help='include media/file_cache directory sizes')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    s = get('/api/status', params={'sizes': args.sizes or None})

    if args.as_json:
        print(json.dumps(s, indent=2))
        return

    def or_dash(v: str | None) -> str:
        return v if v is not None else '-'

    def with_size(path: str | None, size: int | None) -> str:
        return f'{or_dash(path)}  ({_fmt_size(size)})' if size is not None else or_dash(path)

    print(f'pxt_version     {s["pxt_version"]}')
    print(f'daemon_pid      {s["pid"]}')
    print(f'daemon_started  {s["started_at"]}')
    print(f'home            {or_dash(s["home"])}')
    print(f'db_url          {or_dash(s["db_url"])}')
    print(f'media_dir       {with_size(s["media_dir"], s["media_size_bytes"])}')
    print(f'file_cache_dir  {with_size(s["file_cache_dir"], s["file_cache_size_bytes"])}')
    print(f'total_tables    {s["total_tables"]}')
    print(f'total_errors    {s["total_errors"]}')
