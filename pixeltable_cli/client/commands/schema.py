"""Create/update tables from a class-based schema file.

Only the safe subset of the `pxt schema update` spec is implemented: missing tables are created and
existing ones are validated against their model (via the model machinery's idempotent create). There is
no diffing or in-place migration of existing tables yet; a model that doesn't match its existing table is
an error.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..parser import Parser

EPILOG = """\
Examples:
  pxt schema update schema.py my_app
  pxt schema update schema.py my_app --dry-run"""

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
    ap.add_argument('--dry-run', action='store_true', dest='dry_run', help='preview the plan without executing it')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv[1:])
    _update(args.schema, args.target, dry_run=args.dry_run, as_json=args.as_json)


def _error(msg: str) -> None:
    print(f'pxt schema update: {msg}', file=sys.stderr)
    sys.exit(1)


def _update(schema: str, target: str, *, dry_run: bool, as_json: bool) -> None:
    if as_json:
        # keep stdout to the JSON document: suppress pixeltable's console output (connect banner,
        # per-table creation messages)
        os.environ.setdefault('PIXELTABLE_VERBOSITY', '0')

    import pixeltable as pxt
    from pixeltable.catalog.model import TableModelMeta

    schema_path = Path(schema)
    if not schema_path.is_file():
        _error(f'schema file not found: {schema}')

    module_name = schema_path.stem
    spec = importlib.util.spec_from_file_location(module_name, schema_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        _error(f'error loading {schema}: {e}')

    # the schema's model base(s): a base carries __registered_models__ as its own class attribute, whereas
    # the models defined on it merely inherit it
    bases = [
        v for v in vars(module).values() if isinstance(v, TableModelMeta) and '__registered_models__' in v.__dict__
    ]
    if len(bases) == 0:
        _error(f'no model_base() found in {schema}')

    models = [model for base in bases for model in base.__registered_models__.values()]
    binding_root = bases[0]._normalize_binding_root(target)

    def tbl_exists(model: Any) -> bool:
        try:
            return pxt.get_table(f'{binding_root}{model.__table_spec__["name"]}', if_not_exists='ignore') is not None
        except pxt.Error:
            # a missing target directory also reads as a nonexistent table
            return False

    entries: list[dict[str, str]] = []
    for model in models:
        path = f'{binding_root}{model.__table_spec__["name"]}'
        action = 'exists' if tbl_exists(model) else ('create' if dry_run else 'created')
        entries.append({'path': path, 'action': action})
    pending = [e for e in entries if e['action'] != 'exists']

    if not dry_run:
        if target != '':
            pxt.create_dir(target, parents=True, if_exists='ignore')
        try:
            # also validates each model against its table when the table already exists
            for base in bases:
                base.create_all(target)
        except pxt.Error as e:
            _error(str(e))

    if as_json:
        print(json.dumps({'target': target, 'tables': entries}, indent=2))
    else:
        for e in entries:
            print(f'{e["action"]:8s} {e["path"]}')
        if dry_run and len(pending) > 0:
            print(f'{len(pending)} table(s) pending; run without --dry-run to apply')
    if dry_run and len(pending) > 0:
        sys.exit(2)
