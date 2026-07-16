"""Create/update tables from a class-based schema file.

Only the safe subset of the `pxt schema update` spec is implemented: missing tables are created and
existing ones are validated against their model (via the model machinery's idempotent create). There is
no diffing, in-place migration, or dry-run yet; a model that doesn't match its existing table is an error.

The work runs on the daemon (which already has pixeltable loaded); the client stays import-light and only
forwards the schema file's absolute path.
"""

import json
import sys
from pathlib import Path

from ..http import post
from ..parser import Parser

EPILOG = """\
Examples:
  pxt schema update schema.py my_app"""

VERBS = ['update']


def run(argv: list[str]) -> None:
    if len(argv) == 0 or argv[0] in ('-h', '--help'):
        print('usage: pxt schema <verb> [args...]\n\nverbs:\n  update  create missing tables from a schema file')
        sys.exit(0 if len(argv) > 0 else 2)
    verb = argv[0]
    if verb not in VERBS:
        print(f'pxt schema: unknown verb: {verb} (available: {", ".join(VERBS)})', file=sys.stderr)
        sys.exit(2)

    ap = Parser(prog='pxt schema update', epilog=EPILOG)
    ap.add_argument('schema', help='path to a Python file defining a class-based schema')
    ap.add_argument('target', help='catalog directory to create the tables under')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv[1:])
    _update(args.schema, args.target, as_json=args.as_json)


def _update(schema: str, target: str, *, as_json: bool) -> None:
    schema_path = Path(schema)
    if not schema_path.is_file():
        print(f'pxt schema update: schema file not found: {schema}', file=sys.stderr)
        sys.exit(1)

    resp = post('/api/schema/update', {'schema_path': str(schema_path.resolve()), 'target': target})

    if as_json:
        print(json.dumps(resp, indent=2))
        return

    for e in resp['tables']:
        print(f'{e["action"]:8s} {e["path"]}')
