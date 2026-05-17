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
    val: float = n
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if val < 1024:
            return f'{val:.1f}{unit}' if unit != 'B' else f'{int(val)}B'
        val /= 1024
    return f'{val:.1f}PB'


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli status', epilog=EPILOG)
    ap.add_argument('--sizes', action='store_true', help='include media/file_cache directory sizes')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    s = get('/pcli/v0/status' + ('?sizes=1' if args.sizes else ''))

    if args.as_json:
        print(json.dumps(s, indent=2))
        return

    def or_dash(v: str | None) -> str:
        return v if v is not None else '-'

    print(f'pxt_version     {s["pxt_version"]}')
    print(f'daemon_pid      {s["pid"]}')
    print(f'daemon_started  {s["started_at"]}')
    print(f'home            {or_dash(s["home"])}')
    print(f'db_url          {or_dash(s["db_url"])}')
    print(f'media_dir       {or_dash(s["media_dir"])}  ({_fmt_size(s["media_size_bytes"])})')
    print(f'file_cache_dir  {or_dash(s["file_cache_dir"])}  ({_fmt_size(s["file_cache_size_bytes"])})')
    print(f'total_tables    {s["total_tables"]}')
    print(f'total_errors    {s["total_errors"]}')
