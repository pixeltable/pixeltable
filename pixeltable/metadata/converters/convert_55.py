import copy
import logging
from typing import Any
from uuid import UUID

import sqlalchemy as sql

import pixeltable.exceptions as excs
from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md, convert_table_schema_version_md
from pixeltable.metadata.schema import Dir, Table, TableSchemaVersion, TableStatement

_logger = logging.getLogger(__name__)

_DROP_TABLE = TableStatement.DROP_TABLE.value


@register_converter(version=55)
def _(conn: sql.Connection) -> None:
    """
    Changes in version 56:
    - Identifiers (directory, table, column and index names) are case-insensitive, and are stored case-folded.

    Every stored name is rewritten to its folded (lower-case) form. The converter validates first and writes
    nothing if validation fails, so a failed upgrade leaves the database exactly as it was: names must be ASCII,
    and folding must not make two names in the same namespace collide.
    """
    _validate(conn)
    _fold_dirs(conn)
    convert_table_md(conn, table_md_updater=_fold_table_md, substitution_fn=_fold_column_ref_by_name)
    convert_table_schema_version_md(conn, schema_column_updater=_fold_schema_column)
    _log_newly_reserved_names(conn)


def _fold(name: str) -> str:
    return name.lower()


# --- validation ---


def _validate(conn: sql.Connection) -> None:
    """Collect every problem before aborting, so the user sees the full list rather than fixing them one at a time."""
    # one scan for the dropped-table exclusion, shared by all three conflict checks
    live_tbl_ids = _live_tbl_ids(conn)
    problems: list[str] = [
        *_non_ascii_names(conn),
        *_dir_entry_conflicts(conn, live_tbl_ids),
        *_column_conflicts(conn, live_tbl_ids),
        *_index_conflicts(conn, live_tbl_ids),
    ]
    if len(problems) > 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_SCHEMA,
            'Pixeltable identifiers are now case-insensitive and must be ASCII, but this database contains names '
            'that cannot be converted:\n'
            + '\n'.join(f'  - {p}' for p in problems)
            + '\nPlease resolve these with the previous Pixeltable version, then upgrade again.',
        )


def _non_ascii_names(conn: sql.Connection) -> list[str]:
    """Non-ASCII names, over *all* rows -- dropped tables included.

    A dropped table's columns are still reconstructed when its pending drop is rolled forward, and a
    DROP_TABLE statement cannot be aborted, so a name left unconvertible there would fail every retry.
    """
    problems: list[str] = []
    for dir_id, dir_md in conn.execute(sql.select(Dir.id, Dir.md)):
        name = dir_md['name']
        if not name.isascii():
            problems.append(f'directory {name!r} (id={dir_id}) contains non-ASCII characters')
    for tbl_id, tbl_md in conn.execute(sql.select(Table.id, Table.md)):
        if not tbl_md['name'].isascii():
            problems.append(f'table {tbl_md["name"]!r} (id={tbl_id}) contains non-ASCII characters')
        for idx_md in tbl_md['index_md'].values():
            if not idx_md['name'].isascii():
                problems.append(f'index {idx_md["name"]!r} of table id={tbl_id} contains non-ASCII characters')
    for tbl_id, schema_version, sv_md in conn.execute(
        sql.select(TableSchemaVersion.tbl_id, TableSchemaVersion.schema_version, TableSchemaVersion.md)
    ):
        for schema_col in sv_md['columns'].values():
            name = schema_col['name']
            if name is not None and not name.isascii():
                problems.append(
                    f'column {name!r} of table id={tbl_id}, schema version {schema_version} '
                    f'contains non-ASCII characters'
                )
    return problems


def _live_tbl_ids(conn: sql.Connection) -> set[UUID]:
    """Table ids that are visible in the catalog.

    Dropped tables are excluded from every conflict check: they keep their md.name but no user can see or rename
    them, so a long-dropped 'Foo' must not abort the upgrade by colliding with a live 'foo'. Their names are still
    folded -- an interrupted drop is resumed by reloading exactly those rows.
    """
    return {
        tbl_id
        for tbl_id, dir_id, tbl_md in conn.execute(sql.select(Table.id, Table.dir_id, Table.md))
        if dir_id is not None and tbl_md.get('pending_stmt') != _DROP_TABLE
    }


def _dir_entry_conflicts(conn: sql.Connection, live_tbl_ids: set[UUID]) -> list[str]:
    """Two entries of one parent directory whose folded names collide.

    Subdirectories and tables share a single namespace: _get_dir_entry() checks subdirectories and then tables,
    so a directory 'Foo' and a table 'foo' in the same parent are a conflict.
    """
    # parent dir id -> folded name -> spellings
    entries: dict[UUID | None, dict[str, list[str]]] = {}
    for parent_id, dir_md in conn.execute(sql.select(Dir.parent_id, Dir.md)):
        entries.setdefault(parent_id, {}).setdefault(_fold(dir_md['name']), []).append(f'directory {dir_md["name"]!r}')
    for tbl_id, dir_id, tbl_md in conn.execute(sql.select(Table.id, Table.dir_id, Table.md)):
        if tbl_id not in live_tbl_ids:
            continue
        entries.setdefault(dir_id, {}).setdefault(_fold(tbl_md['name']), []).append(f'table {tbl_md["name"]!r}')

    problems: list[str] = []
    for parent_id, by_folded in entries.items():
        for folded, names in by_folded.items():
            if len(names) > 1:
                parent = 'the root directory' if parent_id is None else f'directory id={parent_id}'
                problems.append(f'{parent} contains {", ".join(sorted(names))}, which all become {folded!r}')
    return problems


def _column_conflicts(conn: sql.Connection, live_tbl_ids: set[UUID]) -> list[str]:
    """Columns of one schema version whose folded names collide.

    Every schema version is checked, not only the current one: snapshots and time-travel queries resolve against
    historical versions.
    """
    problems: list[str] = []
    for tbl_id, schema_version, sv_md in conn.execute(
        sql.select(TableSchemaVersion.tbl_id, TableSchemaVersion.schema_version, TableSchemaVersion.md)
    ):
        if tbl_id not in live_tbl_ids:
            continue
        by_folded: dict[str, list[str]] = {}
        for schema_col in sv_md['columns'].values():
            name = schema_col['name']
            if name is not None:
                by_folded.setdefault(_fold(name), []).append(name)
        for folded, names in by_folded.items():
            if len(names) > 1:
                spellings = ', '.join(repr(n) for n in sorted(names))
                problems.append(
                    f'table id={tbl_id}, schema version {schema_version} has columns {spellings}, which all become '
                    f'{folded!r}; a conflict in a historical version can only be resolved by dropping the table, '
                    f'or the snapshots that reach that version'
                )
    return problems


def _index_conflicts(conn: sql.Connection, live_tbl_ids: set[UUID]) -> list[str]:
    """Indexes of one table whose folded names collide.

    Index and column names are separate namespaces (idxs_by_name vs cols_by_name), so a column/index collision is
    not a conflict.
    """
    problems: list[str] = []
    for tbl_id, tbl_md in conn.execute(sql.select(Table.id, Table.md)):
        if tbl_id not in live_tbl_ids:
            continue
        by_folded: dict[str, list[str]] = {}
        for idx_md in tbl_md['index_md'].values():
            by_folded.setdefault(_fold(idx_md['name']), []).append(idx_md['name'])
        for folded, names in by_folded.items():
            if len(names) > 1:
                spellings = ', '.join(repr(n) for n in sorted(names))
                problems.append(f'table id={tbl_id} has indexes {spellings}, which all become {folded!r}')
    return problems


# --- conversion ---


def _fold_dirs(conn: sql.Connection) -> None:
    for dir_id, dir_md in conn.execute(sql.select(Dir.id, Dir.md)):
        updated_md = copy.deepcopy(dir_md)
        updated_md['name'] = _fold(dir_md['name'])
        if updated_md != dir_md:
            conn.execute(sql.update(Dir).where(Dir.id == dir_id).values(md=updated_md))


def _fold_table_md(tbl_md: dict, _tbl_id: UUID) -> None:
    tbl_md['name'] = _fold(tbl_md['name'])
    for idx_md in tbl_md['index_md'].values():
        idx_md['name'] = _fold(idx_md['name'])
    view_md = tbl_md.get('view_md')
    iterator_call = None if view_md is None else view_md.get('iterator_call')
    if iterator_call is not None and iterator_call.get('outputs') is not None:
        # the keys are the component view's column names and must match them exactly; each entry's 'orig_name' is
        # the field the iterator yields at runtime and stays as it is
        iterator_call['outputs'] = {_fold(name): output for name, output in iterator_call['outputs'].items()}


def _fold_schema_column(schema_col: dict) -> None:
    if schema_col['name'] is not None:
        schema_col['name'] = _fold(schema_col['name'])


def _fold_column_ref_by_name(k: str | None, v: Any) -> tuple[str | None, Any] | None:
    """Fold the name of any persisted ColumnRefByName.

    ColumnRefByName is a creation-time placeholder that is substituted before storage, so this should find nothing;
    it is cheap insurance against a stored one.
    """
    if isinstance(v, dict) and v.get('_classname') == 'ColumnRefByName' and isinstance(v.get('name'), str):
        updated = dict(v)
        updated['name'] = _fold(updated['name'])
        return k, updated
    return None


def _log_newly_reserved_names(conn: sql.Connection) -> None:
    """Warn about columns whose folded name lands in the reserved set.

    Such a column keeps loading (the reserved check runs at creation, not at load), but it can no longer be
    re-created or replaced under that name.
    """
    from pixeltable.catalog.globals import is_system_column_name

    offenders: set[str] = set()
    for tbl_id, sv_md in conn.execute(sql.select(TableSchemaVersion.tbl_id, TableSchemaVersion.md)):
        for schema_col in sv_md['columns'].values():
            name = schema_col['name']
            if name is not None and is_system_column_name(name):
                offenders.add(f'{name!r} (table id={tbl_id})')
    if len(offenders) > 0:
        _logger.warning(
            'The following column names are reserved in Pixeltable once case-folded; they still load, but can no '
            'longer be re-created or replaced: ' + ', '.join(sorted(offenders))
        )
