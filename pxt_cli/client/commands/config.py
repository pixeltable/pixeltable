import json

from ..http import get
from ..parser import Parser

EPILOG = """\
Examples:
  pxt config                                # every documented setting
  pxt config -v                             # include description and expected type
  pxt config --section openai               # one section
  pxt config --source env                   # only values set via env var
  pxt config --source file                  # only values set in config.toml
  pxt config --source unset                 # only values that fall back to defaults
  pxt config --json

Notes:
  Source 'env' means an environment variable (or programmatic override) supplies the value.
  Source 'file' means the value comes from config.toml only.
  Source 'unset' means neither layer carries the value; pixeltable falls back to its own default.
  Credential values (api_key, api_token, api_secret, auth_token) show '<redacted>' when set;
  use the source field to tell set from unset for sensitive keys."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt config', epilog=EPILOG)
    ap.add_argument('--section', help='filter to one section (e.g. openai)')
    ap.add_argument('--source', choices=['env', 'file', 'unset'], help='filter by where the value comes from')
    ap.add_argument('-v', '--verbose', action='store_true', help='include description and expected type for each entry')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = get('/api/config')
    entries = resp['entries']
    if args.section is not None:
        entries = [e for e in entries if e['section'] == args.section]
    if args.source is not None:
        entries = [e for e in entries if e['source'] == args.source]

    if args.as_json:
        print(json.dumps({'config_file': resp['config_file'], 'entries': entries}, indent=2))
        return

    print(f'config_file  {resp["config_file"]}')

    set_entries = [e for e in entries if e['source'] != 'unset']
    unset_entries = [e for e in entries if e['source'] == 'unset']

    # With -v, descriptions land on a continuation line under each entry. To keep that
    # readable when nothing is set, list unset entries in the same table form (value '-')
    # instead of collapsing them into the "not set:" one-liner used in non-verbose mode.
    if args.verbose:
        rows = [*set_entries, *unset_entries]
        if len(rows) > 0:
            key_width = max(len(f'{e["section"]}.{e["key"]}') for e in rows)
            src_width = max(len(e['source']) for e in rows) + 2
            for e in rows:
                key = f'{e["section"]}.{e["key"]}'
                src = f'[{e["source"]}]'
                value = e['value'] if e['value'] is not None else '-'
                print(f'{key.ljust(key_width)}  {src.ljust(src_width)}  {value}')
                print(f'    {e["description"]} ({e["expected_type"]})')
        return

    if len(set_entries) > 0:
        key_width = max(len(f'{e["section"]}.{e["key"]}') for e in set_entries)
        src_width = max(len(e['source']) for e in set_entries) + 2  # +2 for surrounding []
        for e in set_entries:
            key = f'{e["section"]}.{e["key"]}'
            src = f'[{e["source"]}]'
            print(f'{key.ljust(key_width)}  {src.ljust(src_width)}  {e["value"]}')

    if len(unset_entries) > 0:
        if len(set_entries) > 0:
            print()
        names = ', '.join(f'{e["section"]}.{e["key"]}' for e in unset_entries)
        print(f'not set: {names}')
