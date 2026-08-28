import copy
from uuid import UUID

import sqlalchemy as sql

import pixeltable.exceptions as excs
from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md, convert_table_schema_version_md
from pixeltable.metadata.schema import Dir, Table, TableSchemaVersion


@register_converter(version=55)
def _(conn: sql.Connection) -> None:
    """
    Changes in version 56:
    - Identifiers (directory, table, column and index names) are case-insensitive, and are stored case-folded.

    Every stored name is rewritten to its folded (lower-case) form. The converter validates first and writes
    nothing if validation fails, so a failed upgrade leaves the database exactly as it was. A name blocks the
    upgrade if it is not ASCII, if folding makes it collide with another in the same namespace, or if folding
    lands it in Pixeltable's reserved set (`Count` -> `count`), which would leave a schema that still loads but
    can no longer be re-created.

    Dropped tables are folded like any other, but excluded from the collision and reserved-name checks: their row
    lingers until the pending drop is finalized, and a name nobody can see must not block an upgrade.
    """
    _validate(conn)
    _fold_dirs(conn)
    convert_table_md(conn, table_md_updater=_fold_table_md)
    convert_table_schema_version_md(conn, schema_column_updater=_fold_schema_column)


def _fold(name: str) -> str:
    return name.lower()


# --- validation ---


def _validate(conn: sql.Connection) -> None:
    """Collect every problem before aborting, so the user sees the full list rather than fixing them one at a time."""
    problems: list[str] = [
        *_non_ascii_names(conn),
        *_dir_entry_conflicts(conn),
        *_column_conflicts(conn),
        *_index_conflicts(conn),
        *_reserved_names(conn),
    ]
    if len(problems) > 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_SCHEMA,
            'Pixeltable identifiers are now case-insensitive, and must be ASCII and not reserved, but this '
            'database contains names that cannot be converted:\n'
            + '\n'.join(f'  - {p}' for p in problems)
            + '\nPlease resolve these with the previous Pixeltable version, then upgrade again.',
        )


def _non_ascii_names(conn: sql.Connection) -> list[str]:
    """Names that are no longer legal identifiers."""
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


def _dir_entry_conflicts(conn: sql.Connection) -> list[str]:
    """Two entries of one parent directory whose folded names collide.

    Subdirectories and tables share a single namespace: _get_dir_entry() checks subdirectories and then tables,
    so a directory 'Foo' and a table 'foo' in the same parent are a conflict.

    Dropped tables are skipped, here and in the other conflict checks: their row lingers with dir_id NULL until
    the pending drop is finalized, and a name nobody can see must not abort the upgrade.
    """
    # parent dir id -> folded name -> spellings
    entries: dict[UUID | None, dict[str, list[str]]] = {}
    for parent_id, dir_md in conn.execute(sql.select(Dir.parent_id, Dir.md)):
        entries.setdefault(parent_id, {}).setdefault(_fold(dir_md['name']), []).append(f'directory {dir_md["name"]!r}')
    for dir_id, tbl_md in conn.execute(sql.select(Table.dir_id, Table.md).where(Table.dir_id.is_not(None))):
        entries.setdefault(dir_id, {}).setdefault(_fold(tbl_md['name']), []).append(f'table {tbl_md["name"]!r}')

    problems: list[str] = []
    for parent_id, by_folded in entries.items():
        for folded, names in by_folded.items():
            if len(names) > 1:
                parent = 'the root directory' if parent_id is None else f'directory id={parent_id}'
                problems.append(f'{parent} contains {", ".join(sorted(names))}, which all become {folded!r}')
    return problems


def _column_conflicts(conn: sql.Connection) -> list[str]:
    """Columns of one schema version whose folded names collide.

    Every schema version is checked, not only the current one: snapshots and time-travel queries resolve against
    historical versions.
    """
    problems: list[str] = []
    stmt = (
        sql.select(TableSchemaVersion.tbl_id, TableSchemaVersion.schema_version, TableSchemaVersion.md)
        .join(Table, Table.id == TableSchemaVersion.tbl_id)
        .where(Table.dir_id.is_not(None))
    )
    for tbl_id, schema_version, sv_md in conn.execute(stmt):
        by_folded: dict[str, list[str]] = {}
        for schema_col in sv_md['columns'].values():
            name = schema_col['name']
            if name is not None:
                by_folded.setdefault(_fold(name), []).append(name)
        for folded, names in by_folded.items():
            if len(names) > 1:
                spellings = ', '.join(repr(name) for name in sorted(names))
                problems.append(
                    f'table id={tbl_id}, schema version {schema_version} has columns {spellings}, which all become '
                    f'{folded!r}; a conflict in a historical version can only be resolved by dropping the table, '
                    f'or the snapshots that reach that version'
                )
    return problems


def _index_conflicts(conn: sql.Connection) -> list[str]:
    """Indexes of one table whose folded names collide.

    Index and column names are separate namespaces (idxs_by_name vs cols_by_name), so a column/index collision is
    not a conflict.
    """
    problems: list[str] = []
    for tbl_id, tbl_md in conn.execute(sql.select(Table.id, Table.md).where(Table.dir_id.is_not(None))):
        by_folded: dict[str, list[str]] = {}
        for idx_md in tbl_md['index_md'].values():
            by_folded.setdefault(_fold(idx_md['name']), []).append(idx_md['name'])
        for folded, names in by_folded.items():
            if len(names) > 1:
                spellings = ', '.join(repr(name) for name in sorted(names))
                problems.append(f'table id={tbl_id} has indexes {spellings}, which all become {folded!r}')
    return problems


def _reserved_names(conn: sql.Connection) -> list[str]:
    """Column names that land in Pixeltable's reserved set once folded.

    `Count` becomes `count`, which Column.validate_name() rejects, so the column can no longer be re-created or
    replaced -- the schema stops being reproducible even though it still loads. Only the current schema version is
    checked: a historical one is never re-created, and could not be renamed anyway.
    """
    from keyword import iskeyword

    from pixeltable.catalog.globals import is_system_column_name

    current_versions = {
        tbl_id: tbl_md['current_schema_version']
        for tbl_id, tbl_md in conn.execute(sql.select(Table.id, Table.md).where(Table.dir_id.is_not(None)))
    }
    problems: list[str] = []
    stmt = sql.select(TableSchemaVersion.tbl_id, TableSchemaVersion.schema_version, TableSchemaVersion.md)
    for tbl_id, schema_version, sv_md in conn.execute(stmt):
        if current_versions.get(tbl_id) != schema_version:
            continue
        for schema_col in sv_md['columns'].values():
            name = schema_col['name']
            if name is None:
                continue
            folded = _fold(name)
            if is_system_column_name(folded) or iskeyword(folded):
                problems.append(
                    f'column {name!r} of table id={tbl_id} becomes {folded!r}, which is a reserved name in '
                    f'Pixeltable; rename it with the previous Pixeltable version'
                )
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
