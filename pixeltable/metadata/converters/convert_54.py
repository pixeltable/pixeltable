import copy
from typing import Any
from uuid import UUID

import sqlalchemy as sql

from pixeltable import exceptions as excs
from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import Table, TableSchemaVersion


@register_converter(version=54)
def _(conn: sql.Connection) -> None:
    """
    Changes in version 55:
    - A function that can only be serialized by pickling its body (an Expr.apply() result or a locally-defined
      @pxt.udf, ie a CallableFunction without a fully-qualified path) can no longer be persisted into catalog
      metadata, because a pickled body may not load after a Python version change. The legacy 'functions' table
      that stored such bodies out-of-line is dropped.

    A stored database may already contain such functions, either inline (a function dict with a 'binary' key) or
    as a reference into the 'functions' table (a function dict with an 'id' key). These cannot be carried forward,
    so this converter aborts if it finds any: raising rolls back the transaction and leaves the database at
    version 54. Persisted function dicts can appear in Table.md (embedding-index init_args, view predicates and
    iterator args) and in TableSchemaVersion.md (computed-column value_exprs).

    Only each table's current schema version is scanned. Dropping an offending column (the fix the abort message
    directs the user to) advances the schema version, and prior schema versions are immutable history that a
    revert may reload but that the live schema no longer references.
    """
    _strip_stored_proxy_columns(conn)

    tbl_names: dict[UUID, str] = {}
    tbl_mds: dict[UUID, dict] = {}
    current_schema_version: dict[UUID, int] = {}
    for row in conn.execute(sql.select(Table.id, Table.md)):
        tbl_id, table_md = row[0], row[1]
        tbl_names[tbl_id] = table_md.get('name') or str(tbl_id)
        tbl_mds[tbl_id] = table_md
        current_schema_version[tbl_id] = table_md['current_schema_version']

    offending: set[str] = set()

    # computed-column value_exprs live in the current schema version
    for row in conn.execute(
        sql.select(TableSchemaVersion.tbl_id, TableSchemaVersion.schema_version, TableSchemaVersion.md)
    ):
        tbl_id, schema_version, sv_md = row[0], row[1], row[2]
        if schema_version != current_schema_version.get(tbl_id):
            continue
        for schema_col in sv_md['columns'].values():
            if _contains_pickled_fn(schema_col.get('value_expr')):
                col_name = schema_col.get('name')
                offending.add(f'{tbl_names[tbl_id]}.{col_name}' if col_name is not None else tbl_names[tbl_id])

    # embedding-index init_args, and view predicates / iterator args, live in Table.md
    for tbl_id, table_md in tbl_mds.items():
        for idx_md in (table_md.get('index_md') or {}).values():
            if _contains_pickled_fn(idx_md):
                offending.add(f'{tbl_names[tbl_id]}: embedding index {idx_md.get("name")!r}')
        if _contains_pickled_fn(table_md.get('view_md')):
            offending.add(f'{tbl_names[tbl_id]}: view definition')

    if len(offending) > 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            'Cannot upgrade the Pixeltable metadata: the following computed columns, view definitions, or '
            'embedding indexes\nuse a function created with `.apply()` or defined as a local function (eg, in a '
            'notebook):\n'
            f'  {", ".join(sorted(offending))}\n'
            'Such functions are stored as a pickled body that may not load after a Python version change and are '
            'no longer supported.\nDrop those columns or indexes, or recreate them using a module-level '
            '`@pxt.udf`, then retry the upgrade.',
        )

    # no pickled functions remain; drop the legacy out-of-line functions table
    conn.execute(sql.text('DROP TABLE IF EXISTS functions'))


def _strip_stored_proxy_columns(conn: sql.Connection) -> None:
    """Remove orphaned 'stored proxy' columns left behind by the removed external-store/LabelStudio integration.

    That integration materialized a media column for export by adding a proxy column whose value_expr pickles an
    identity function; it was removed without a converter, so those columns linger in existing metadata. They are
    system columns (no user-facing name) that no user can drop, and their pickled function would trip the ban
    below. A proxy has no dependents, so removing it from every schema version and from the table's column set is
    safe and leaves the table and its user columns intact.
    """
    proxy_col_ids: dict[UUID, set[str]] = {}
    for row in conn.execute(sql.select(TableSchemaVersion.tbl_id, TableSchemaVersion.md)):
        tbl_id, sv_md = row[0], row[1]
        for col_id, schema_col in sv_md['columns'].items():
            # A proxy is a system column (no name) whose value_expr is a FunctionCall applying the pickled identity
            # function. Match the call's own function, not the whole value_expr: the latter also carries the
            # referenced column's metadata, whose pickled function would produce a false positive.
            value_expr = schema_col.get('value_expr')
            fn = value_expr.get('fn') if isinstance(value_expr, dict) else None
            if schema_col.get('name') is None and _contains_pickled_fn(fn):
                proxy_col_ids.setdefault(tbl_id, set()).add(col_id)

    if len(proxy_col_ids) == 0:
        return

    for row in conn.execute(
        sql.select(TableSchemaVersion.tbl_id, TableSchemaVersion.schema_version, TableSchemaVersion.md)
    ):
        tbl_id, schema_version, sv_md = row[0], row[1], row[2]
        col_ids = proxy_col_ids.get(tbl_id)
        if col_ids is None or not any(col_id in sv_md['columns'] for col_id in col_ids):
            continue
        updated_md = copy.deepcopy(sv_md)
        for col_id in col_ids:
            updated_md['columns'].pop(col_id, None)
        conn.execute(
            sql.update(TableSchemaVersion)
            .where(TableSchemaVersion.tbl_id == tbl_id)
            .where(TableSchemaVersion.schema_version == schema_version)
            .values(md=updated_md)
        )

    for row in conn.execute(sql.select(Table.id, Table.md)):
        tbl_id, table_md = row[0], row[1]
        col_ids = proxy_col_ids.get(tbl_id)
        if col_ids is None or not any(col_id in table_md['column_md'] for col_id in col_ids):
            continue
        updated_md = copy.deepcopy(table_md)
        for col_id in col_ids:
            updated_md['column_md'].pop(col_id, None)
        conn.execute(sql.update(Table).where(Table.id == tbl_id).values(md=updated_md))


def _contains_pickled_fn(md: Any) -> bool:
    """True if md contains a serialized function whose body is stored by pickle.

    A serialized function is a dict with a '_classpath' key. The pickle-backed forms carry either a 'binary' key
    (the body inlined by value) or an 'id' key (a reference into the legacy 'functions' table).
    """
    if isinstance(md, dict):
        if '_classpath' in md and ('binary' in md or 'id' in md):
            return True
        return any(_contains_pickled_fn(v) for v in md.values())
    if isinstance(md, list):
        return any(_contains_pickled_fn(v) for v in md)
    return False
