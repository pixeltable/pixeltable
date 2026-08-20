import json
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

from ...schema_types import DiffResolution, OpStatus, SchemaChangeOp, SchemaPlan, drop_table_op
from ...utils import PxtPath
from ..confirm import confirm_or_exit
from ..parser import Parser
from ..utils import post_request

# a working schema file: written verbatim by 'pxt schema example', and shown indented in every verb's epilog,
# because otherwise the shape of a model file has to be guessed
_BRIEF_EXAMPLE_SCHEMA = """\
from __future__ import annotations  # required to declare a model on Python 3.14+

import pixeltable as pxt
import pixeltable.functions as pxtf

TableModel = pxt.model_base()


class Docs(TableModel, name='docs'):
    title: pxt.String                         # a stored column
    body: pxt.String | None                   # a stored column that may be null
    title_upper = pxtf.string.upper(title)    # a computed column: an assignment, not an annotation


class Titled(TableModel, name='titled', base=Docs.where(Docs.title != '')):
    headline = Docs.title_upper + '!'         # a view of Docs, filtered by its base= query
"""

_EXAMPLE_SCHEMA = '''\
"""Pixeltable schema, written by 'pxt schema example'.

Every construct the schema DSL supports appears below; delete what you do not need.
'pxt schema example --brief' prints a minimal starting point instead.

'pxt schema diff FILE TARGET' reports what applying this would change; 'pxt schema update' applies it.
Building an application with Pixeltable? The agent skill carries the full API:
    npx skills add pixeltable/pixeltable-skill
"""

from __future__ import annotations  # required to declare a model on Python 3.14+

import pixeltable as pxt
import pixeltable.functions as pxtf

TableModel = pxt.model_base()


class Docs(TableModel, name='docs'):
    """One model becomes one table, named by name=."""

    # an annotation declares a stored column
    doc_id: pxt.Int
    title: pxt.String
    body: pxt.String | None
    published: pxt.Timestamp | None
    tags: pxt.Json | None
    rating: pxt.Float | None
    is_draft: pxt.Bool | None
    embedding: pxt.Array[(384,), pxt.Float] | None
    source: pxt.Document | None  # a media column takes a local path or a URL on insert

    # an assignment declares a computed column, evaluated on insert and on update
    title_upper = pxtf.string.upper(title)
    summary = pxtf.string.slice(body, 0, 80)

    # an embedding index makes a column searchable by similarity
    __indexes__ = [
        pxt.EmbeddingIndex(
            body,
            embedding=pxtf.huggingface.sentence_transformer.using(model_id='sentence-transformers/all-MiniLM-L6-v2'),
            name='body_idx',
        )
    ]


class Recordings(TableModel, name='recordings'):
    """The other media types, and the column properties an annotation cannot express."""

    recording_id: pxt.Int
    cover: pxt.Image | None
    narration: pxt.Audio | None
    clip: pxt.Video | None

    thumbnail = pxt.Column(value=cover.rotate(90), stored=False)  # computed on read, never stored
    scan = pxt.Column(type=pxt.Image, media_validation='on_read', comment='validated on read, not on insert')


class Titled(TableModel, name='titled', base=Docs.where(Docs.title != '')):
    """A view: base= is a query over another model. Its rows follow the base and are not inserted directly."""

    headline = Docs.title_upper + '!'


class Sentences(
    TableModel,
    name='sentences',
    base=Docs,
    iterator=pxtf.string.string_splitter(Docs.body, separators='sentence'),
):
    """A component view: one row per item the iterator produces from each base row."""

    # 'text' is an output column of the iterator, not declared here
    length = pxtf.string.len(text)  # type: ignore[name-defined]  # noqa: F821
'''

_SCHEMA_FILE = f"""\
Schema file:
  A Python module defining one or more models on a pxt.model_base(). This is the minimal form;
  'pxt schema example' writes one covering every supported construct.

{textwrap.indent(_BRIEF_EXAMPLE_SCHEMA.rstrip(), '    ')}

  Each model becomes one table under TARGET, named by name=."""

DIFF_EPILOG = f"""\
Examples:
  pxt schema diff schema.py my_app                 # what 'schema update' would change
  pxt schema diff schema.py my_app --json          # the same plan, machine-readable
  pxt schema diff schema.py pxt://org:db/prod      # against a hosted database

Output:
  + <path>      table will be created        + <column>   will be added
  ~ <path>      table will be migrated       - <column>   will be dropped
  = <path>      already matches its model
  ! <path>      cannot be migrated in place, or is not declared by the schema
  Each operation is marked safe, DESTRUCTIVE, or UNSUPPORTED.

Exit codes:
  0  the target matches the schema
  2  changes are pending
  1  error: bad arguments, the schema file failed to import, or the daemon is unreachable

Notes:
  Read-only: never creates TARGET, never touches a table.
  Tables under TARGET that no model declares are reported as extras. 'schema update' never
  removes them, so they do not count as pending changes and do not affect the exit code.

{_SCHEMA_FILE}"""

UPDATE_EPILOG = f"""\
Examples:
  pxt schema diff   schema.py my_app                  # review first
  pxt schema update schema.py my_app                  # then apply
  pxt schema update schema.py my_app -n               # dry-run: the plan, applying nothing
  pxt schema update schema.py my_app --allow-destructive -f
  pxt schema update schema.py my_app --json

Exit codes:
  0  the target now matches the schema (including when there was nothing to do)
  2  with -n, changes are pending
  3  refused: the plan is destructive and --allow-destructive was not given, or -f was needed
  1  error: bad arguments, the schema file failed to import, or a table cannot be reconciled

Notes:
  Creates TARGET and any missing tables, and migrates existing ones (adding and dropping columns
  and indexes). Re-running against an unchanged schema does nothing.
  A change that cannot be applied in place -- a kind or iterator mismatch, or a column whose type
  or properties changed -- is reported as UNSUPPORTED and nothing is applied. Adjust the schema
  or the table by hand.
  Dropping a column or an index destroys its data and needs --allow-destructive. Applying is
  all-or-nothing: without that flag a destructive plan applies nothing at all.
  Tables under TARGET that no model declares are left alone; 'schema prune' removes them.
  The daemon imports the schema file, so it must be readable there; the file's own directory is
  added to sys.path, so it can import modules sitting next to it.

{_SCHEMA_FILE}"""

PRUNE_EPILOG = f"""\
Examples:
  pxt schema prune schema.py my_app -n     # list what would be dropped
  pxt schema prune schema.py my_app -f     # drop it
  pxt schema prune schema.py my_app --json

Exit codes:
  0  nothing left to prune
  2  with -n, tables would be dropped
  3  drops were refused: -f is required when there is no terminal to confirm at
  1  error: bad arguments, the schema file failed to import, or a drop failed

Notes:
  Drops every table under TARGET that no model declares. This is irreversible.
  Only tables under TARGET are considered, so nothing elsewhere in the catalog is affected.
  Declared tables are never dropped, and never modified: a full reconcile is 'update' then 'prune'.
  A view is dropped before its base. Prune never force-drops, so a table that something outside the
  pruned set depends on is left in place and the drop fails, naming what depends on it.
  Without -f, confirmation is read from the terminal; non-interactive callers must pass -f.

{_SCHEMA_FILE}"""

EXAMPLE_EPILOG = f"""\
Examples:
  pxt schema example                       # print it
  pxt schema example --brief               # a minimal schema instead
  pxt schema example --out schema.py       # write it, then edit and apply
  pxt schema example --out schema.py && pxt schema update schema.py my_app

Notes:
  The file is a working schema: applying it as-is creates the tables it declares.
  It covers every construct the DSL supports, so there is nothing to look up elsewhere; delete
  whatever the application does not need.

{_SCHEMA_FILE}"""

VERBS = ['diff', 'update', 'prune', 'example']

# exit status: whether the target already matches the schema is reported here, not only in the output
EXIT_IN_AGREEMENT = 0
EXIT_ERROR = 1
EXIT_CHANGES_PENDING = 2
EXIT_REFUSED = 3


class _Rendering(NamedTuple):
    marker: str
    pending: str
    applied: str  # empty for a resolution that is never carried out


_RESOLUTIONS: dict[DiffResolution, _Rendering] = {
    'create': _Rendering('+', 'create', 'created'),
    'update_additive': _Rendering('~', 'update', 'updated'),
    'update_destructive': _Rendering('~', 'update', 'updated'),
    'up_to_date': _Rendering('=', 'no change', 'unchanged'),
    'unsupported': _Rendering('!', 'unsupported', ''),
}

# how an operation's severity reads
_SEVERITY_LABELS = {'additive': 'safe', 'destructive': 'DESTRUCTIVE', 'unsupported': 'UNSUPPORTED'}

# the marker introducing an operation's line; an alter is an in-place change
_OP_MARKERS = {'add': '+', 'drop': '-'}


def run(argv: list[str]) -> None:
    if len(argv) == 0 or argv[0] in ('-h', '--help'):
        print(
            'usage: pxt schema <verb> SCHEMA TARGET [options]\n\nverbs:\n'
            '  diff     show the changes that update would make; exit 2 if any are pending\n'
            '  update   create and migrate the tables the schema declares under TARGET\n'
            '  prune    drop the tables under TARGET that the schema does not declare\n'
            '  example  write a working schema file to start from (takes no SCHEMA/TARGET)\n\n'
            'SCHEMA is a Python file defining models on a pxt.model_base(); TARGET is a catalog\n'
            "directory or a pxt:// URI. Run 'pxt schema example' for a file to start from."
        )
        sys.exit(EXIT_IN_AGREEMENT if len(argv) > 0 else EXIT_ERROR)
    verb = argv[0]
    if verb not in VERBS:
        print(f'pxt schema: unknown verb: {verb} (available: {", ".join(VERBS)})', file=sys.stderr)
        sys.exit(EXIT_ERROR)

    if verb == 'example':
        ap = Parser(prog='pxt schema example', epilog=EXAMPLE_EPILOG, usage_exit_code=EXIT_ERROR)
        ap.add_argument('--out', help='write to this file instead of standard output')
        ap.add_argument('--brief', action='store_true', help='a minimal schema instead of the full one')
        args = ap.parse_args(argv[1:])
        _example(args.out, brief=args.brief)
        return

    epilogs = {'diff': DIFF_EPILOG, 'update': UPDATE_EPILOG, 'prune': PRUNE_EPILOG}
    # a usage error exits EXIT_ERROR, not argparse's 2, which here means that changes are pending
    ap = Parser(prog=f'pxt schema {verb}', epilog=epilogs[verb], usage_exit_code=EXIT_ERROR)
    ap.add_argument('schema', help='path to a Python file defining a class-based schema')
    ap.add_argument('target', help='catalog directory to reconcile against the schema')
    ap.add_argument('--json', action='store_true', dest='as_json')
    if verb in ('update', 'prune'):
        ap.add_argument('-f', '--force', action='store_true', help='skip confirmation')
        ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    if verb == 'update':
        ap.add_argument(
            '--allow-destructive',
            action='store_true',
            dest='allow_destructive',
            help='permit operations that drop a column or index',
        )
    args = ap.parse_args(argv[1:])

    path = Path(args.schema)
    if not path.is_file():
        print(
            f'pxt schema {verb}: schema file not found: {args.schema}\n'
            "run 'pxt schema example' for a file to start from",
            file=sys.stderr,
        )
        sys.exit(EXIT_ERROR)
    schema_file = str(path.resolve())

    if verb == 'diff':
        _diff(schema_file, args.target, as_json=args.as_json)
    elif verb == 'prune':
        _prune(schema_file, args.target, as_json=args.as_json, force=args.force, dry_run=args.dry_run)
    else:
        _update(
            schema_file,
            args.target,
            as_json=args.as_json,
            force=args.force,
            dry_run=args.dry_run,
            allow_destructive=args.allow_destructive,
        )


def _example(out: str | None, *, brief: bool) -> None:
    text = _BRIEF_EXAMPLE_SCHEMA if brief else _EXAMPLE_SCHEMA
    if out is None:
        sys.stdout.write(text)
        return
    Path(out).write_text(text, encoding='utf-8')
    print(f'wrote {out}')


def _diff(schema_file: str, catalog_dir: PxtPath, *, as_json: bool) -> None:
    plan = _plan_for(schema_file, catalog_dir)
    _diff_output(plan, as_json=as_json)
    sys.exit(EXIT_IN_AGREEMENT if plan['in_agreement'] else EXIT_CHANGES_PENDING)


def _plan_for(schema_file: str, catalog_dir: PxtPath) -> SchemaPlan:
    plan: SchemaPlan = post_request('/api/schema/diff', {'schema_file': schema_file, 'catalog_dir': catalog_dir})
    return plan


def _format_plan(plan: SchemaPlan) -> list[str]:
    lines: list[str] = []
    for tbl in plan['tables']:
        rendering = _RESOLUTIONS[tbl['resolution']]
        lines.append(f'{rendering.marker} {tbl["path"]:<24s} {rendering.pending}')
        for op in tbl['ops']:
            lines.append(f'    {_OP_MARKERS.get(op["op"], "~")} {op["description"]}  {_severity_label(op)}')
    for path in plan['extras']:
        lines.append(f'! {path:<24s} extra (not in schema)')

    s = plan['summary']
    updates = s['update_additive'] + s['update_destructive']
    counts = f'{s["create"]} create, {updates} update, {s["up_to_date"]} unchanged, {s["extras"]} extra'
    if s['unsupported'] > 0:
        counts += f', {s["unsupported"]} unsupported'
    lines.append('')
    lines.append(f'Plan: {counts}  |  {s["destructive"]} destructive')
    return lines


def _severity_label(op: SchemaChangeOp) -> str:
    # an unmapped severity prints as itself: a category added later must not read as harmless here
    return _SEVERITY_LABELS.get(op['severity'], op['severity'].upper())


def _prune(schema_file: str, catalog_dir: PxtPath, *, as_json: bool, force: bool, dry_run: bool) -> None:
    plan = _plan_for(schema_file, catalog_dir)
    extras = plan['extras']
    if len(extras) == 0:
        if as_json:
            print(json.dumps({**plan, 'ops': []}, indent=2))
        else:
            print('nothing to prune')
        sys.exit(EXIT_IN_AGREEMENT)

    if dry_run:
        _prune_output(
            {**plan, 'ops': [drop_table_op(p, 'skipped') for p in extras]}, as_json=as_json, verb='would drop'
        )
        sys.exit(EXIT_CHANGES_PENDING)

    def report_refusal() -> None:
        _prune_output(
            {**plan, 'ops': [drop_table_op(p, 'refused') for p in extras]}, as_json=as_json, verb='would drop'
        )

    confirm_or_exit(
        f'drop {len(extras)} table(s) not declared by the schema?',
        force,
        refused_exit_code=EXIT_REFUSED,
        on_refusal=report_refusal,
    )

    resp = post_request('/api/schema/prune', {'schema_file': schema_file, 'catalog_dir': catalog_dir})
    _prune_output(resp, as_json=as_json, verb='dropped')


def _prune_output(plan: SchemaPlan, *, as_json: bool, verb: str) -> None:
    if as_json:
        print(json.dumps(plan, indent=2))
        return
    for op in plan['ops']:
        print(f'{verb} {op["name"]}')


def _update(
    schema_file: str, catalog_dir: PxtPath, *, as_json: bool, force: bool, dry_run: bool, allow_destructive: bool
) -> None:
    if dry_run:
        plan = _plan_for(schema_file, catalog_dir)
        _set_statuses(plan, destructive='skipped', other='skipped')
        _diff_output(plan, as_json=as_json)
        sys.exit(EXIT_IN_AGREEMENT if plan['in_agreement'] else EXIT_CHANGES_PENDING)

    # the plan is read up front only to decide whether to proceed: with destructive operations already permitted
    # and confirmation waived, there is nothing left to decide
    if not (allow_destructive and force):
        _decide_update(schema_file, catalog_dir, as_json=as_json, force=force, allow_destructive=allow_destructive)

    applied = post_request(
        '/api/schema/update',
        {'schema_file': schema_file, 'catalog_dir': catalog_dir, 'allow_destructive': allow_destructive},
    )
    _update_output(applied, as_json=as_json)


def _decide_update(
    schema_file: str, catalog_dir: PxtPath, *, as_json: bool, force: bool, allow_destructive: bool
) -> None:
    """Reports the pending plan and exits, unless applying it is permitted.

    Exits 0 if there is nothing to apply, and 3 if the plan is destructive and that was neither permitted nor
    confirmed. Returning means the plan may be applied; it is advisory, being a separate read from the one the
    apply acts on.
    """
    plan = _plan_for(schema_file, catalog_dir)
    if plan['in_agreement']:
        _update_output(plan, as_json=as_json)
        sys.exit(EXIT_IN_AGREEMENT)

    destructive = plan['summary']['destructive']
    if destructive == 0:
        return
    if not allow_destructive:
        _set_statuses(plan, destructive='refused', other='skipped')
        _diff_output(plan, as_json=as_json)
        print(
            f'pxt schema update: refusing to apply {destructive} destructive operation(s) without --allow-destructive',
            file=sys.stderr,
        )
        sys.exit(EXIT_REFUSED)
    confirm_or_exit(f'apply {destructive} destructive operation(s)?', force, refused_exit_code=EXIT_REFUSED)


def _update_output(plan: SchemaPlan, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(plan, indent=2))
        return
    if plan['in_agreement']:
        print('catalog is up to date')
        return
    for tbl in plan['tables']:
        print(f'{_RESOLUTIONS[tbl["resolution"]].applied:9s} {tbl["path"]}')


def _set_statuses(plan: SchemaPlan, *, destructive: OpStatus, other: OpStatus) -> None:
    """Puts a status on every table and operation of the plan.

    Destructive operations take the destructive status, the rest take the other one, and a table takes the
    destructive status if any of its operations does.
    """
    for tbl in plan['tables']:
        for op in tbl['ops']:
            op['status'] = destructive if op['destructive'] else other
        tbl['status'] = destructive if any(op['destructive'] for op in tbl['ops']) else other


def _diff_output(plan: SchemaPlan, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(plan, indent=2))
        return
    for line in _format_plan(plan):
        print(line)
