from __future__ import annotations

import pixeltable as pxt
from pixeltable import catalog, exceptions as excs
from pixeltable.config import Config
from pixeltable.service.db import db_fingerprint
from pixeltable.utils.app_module import (
    check_report,
    check_udf_references,
    get_model_bases,
    load_app_module,
    udf_source_files,
)
from pixeltable.utils.project import project_fingerprint
from pixeltable_cli.types import CheckReport, SchemaChangeOp, SchemaPlan, TableDiff
from pixeltable_cli.utils import PxtPath

from .definition import TableModelMeta
from .diff import PY_DESTRUCTIVE_HINT

_DESTRUCTIVE_HINT = "Re-run 'pxt schema update' with --allow-destructive to apply these changes."


def schema_diff(app_file: str, catalog_dir: PxtPath) -> SchemaPlan:
    """Return the plan that reconciles the tree under catalog_dir with app_file."""
    bases = _model_bases(app_file)
    _validate_udfs(app_file, bases)
    return _schema_plan(_model_diffs(bases, catalog_dir), app_file, catalog_dir, _blocked_ops(catalog_dir, bases))


def schema_update(app_file: str, catalog_dir: PxtPath, *, allow_destructive: bool = False) -> SchemaPlan:
    """Return the plan needed to reconcile the tree under catalog_dir with app_file."""
    model_bases = _model_bases(app_file)
    _validate_udfs(app_file, model_bases)

    blocked = _blocked_ops(catalog_dir, model_bases)
    if len(blocked) > 0:
        plan = _schema_plan(_model_diffs(model_bases, catalog_dir), app_file, catalog_dir, blocked)
        for tbl in plan.tables:
            tbl.status = 'refused'
            for op in tbl.ops:
                op.status = 'refused'
        return plan

    # only create catalog_dir when it names an in-catalog path; a bare catalog root (eg '' or 'pxt://org:db')
    # has no directory to create
    if len(catalog.Path.parse(catalog_dir, allow_empty_path=True).components) > 0:
        pxt.create_dir(catalog_dir, parents=True, if_exists='ignore')

    applied: list[TableDiff] = []
    for base in model_bases:
        try:
            diffs = base.update_all(catalog_dir, allow_destructive=allow_destructive)
        except excs.Error as e:
            message = e.message
            if e.error_code is excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE:
                # update_all() closes its refusal with instructions for the Python API; a CLI user needs the flag
                message = message.replace(PY_DESTRUCTIVE_HINT, _DESTRUCTIVE_HINT)
            migrated = [d.path for d in applied if d.resolution != 'up_to_date']
            if len(migrated) > 0:
                # the migrations from earlier bases are already committed; name them, so the error doesn't
                # read as though the catalog were untouched
                message += f'\n\nThe following table(s) were already migrated: {", ".join(repr(p) for p in migrated)}.'
            e.args = (message,)
            raise
        applied.extend(diffs.values())

    plan = _schema_plan(applied, app_file, catalog_dir, [])
    for tbl in plan.tables:
        tbl.status = 'skipped' if tbl.resolution == 'up_to_date' else 'applied'
        for op in tbl.ops:
            op.status = 'applied'
    return plan


def schema_prune(app_file: str, catalog_dir: PxtPath) -> SchemaPlan:
    """Drop the tables under catalog_dir that are absent in app_file.

    Returns the plan, with one drop_table operation per dropped table. A view is dropped before its base, so that
    pruning a group of related tables does not depend on the order they are listed in. Nothing is force-dropped:
    a table that something outside the pruned set depends on is left in place and its error is raised.
    If this exits with an error, it may have dropped a partial list of tables.
    """
    plan = schema_diff(app_file, catalog_dir)
    remaining = list(plan.extras)
    dropped: list[PxtPath] = []
    while len(remaining) > 0:
        deferred: list[PxtPath] = []
        blocked_by: excs.Error | None = None
        for pxt_path in remaining:
            try:
                pxt.drop_table(pxt_path, if_not_exists='ignore')
            except excs.Error as e:
                blocked_by = e
                deferred.append(pxt_path)
                continue
            dropped.append(pxt_path)
        if len(deferred) == len(remaining):
            assert blocked_by is not None
            if len(dropped) > 0:
                # the drops so far are already committed; name them, so the error doesn't read as though the
                # catalog were untouched. Augmenting in place keeps the exception's type and fields, and
                # blocked_by.message excludes blocked_by.detail, which must not become part of the message.
                names = ', '.join(repr(pxt_path) for pxt_path in dropped)
                blocked_by.args = (f'{blocked_by.message}\n\nThe following table(s) were already dropped: {names}.',)
            raise blocked_by
        remaining = deferred

    plan.ops = [SchemaChangeOp.drop_table(pxt_path, 'applied') for pxt_path in dropped]
    return plan


def schema_check(app_file: str) -> CheckReport:
    return check_report(app_file, _model_bases(app_file))


def _model_bases(app_file: str) -> list[TableModelMeta]:
    """The model bases in app_file."""
    bases = get_model_bases(load_app_module(app_file, subject='schema file'))
    if len(bases) == 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            f"no model_base() found in {app_file}; run 'pxt schema example' for a file to start from",
        )
    return bases


def _validate_udfs(app_file: str, bases: list[TableModelMeta]) -> None:
    """Refuse a schema that references unresolvable udfs."""
    errors = check_udf_references(bases)
    if len(errors) > 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, '\n'.join([f'{app_file} contains unresolvable udf references:', *errors])
        )


def _model_diffs(bases: list[TableModelMeta], catalog_dir: PxtPath) -> list[TableDiff]:
    """Returns diffs of bases against their corresponding tables in catalog_dir."""
    return [diff for base in bases for diff in base.get_model_diff(catalog_dir).values()]


def _blocked_ops(catalog_dir: PxtPath, bases: list[TableModelMeta]) -> list[SchemaChangeOp]:
    """Return a blocked operation if a hosted target's project lacks a udf source file or holds an older copy."""
    db_path = catalog.Path.parse(catalog_dir, allow_empty_path=True)
    deployed = db_fingerprint(db_path)
    if deployed is None:
        # the target reads the udfs from the files on disk
        return []
    project_root = Config.get().project_root
    assert project_root is not None
    local = project_fingerprint(project_root, Config.get().get_database_config(db_path))
    stale: list[str] = []
    for path in sorted(udf_source_files(bases)):
        # compare the file hashes
        if path not in local.files or local.files[path] == deployed.files.get(path):
            continue
        stale.append(f'{path} changed' if path in deployed.files else f'{path} added')
    if len(stale) == 0:
        return []
    return [SchemaChangeOp.needs_db_update(stale, f'pxt db update {db_path.uri_str}')]


def _schema_plan(diffs: list[TableDiff], app_file: str, catalog_dir: PxtPath, ops: list[SchemaChangeOp]) -> SchemaPlan:
    # a create subsumes the additions that constitute it, so only a migration enumerates operations
    tables = [d.model_copy(update={'ops': []}) if d.resolution in ('create', 'up_to_date') else d for d in diffs]

    # a table's path crosses from the catalog as a plain string
    defined = {_path_key(PxtPath(d.path)) for d in tables}
    return SchemaPlan(
        app_file=app_file,
        catalog_dir=catalog_dir,
        ops=ops,
        tables=tables,
        # extras are excluded from in_agreement: update() never removes them, so their presence is not
        # something it could reconcile
        extras=sorted(p for p in _list_tables(catalog_dir) if _path_key(p) not in defined),
    )


def _path_key(pxt_path: PxtPath) -> tuple[str, ...]:
    """A comparable identity for a table path, so that a pxt:// URI and a bare path denote the same table."""
    return tuple(catalog.Path.parse(pxt_path, allow_empty_path=True).components)


def _list_tables(pxt_path: PxtPath) -> list[PxtPath]:
    try:
        return [PxtPath(p) for p in pxt.list_tables(pxt_path, recursive=True)]
    except excs.NotFoundError:
        return []
