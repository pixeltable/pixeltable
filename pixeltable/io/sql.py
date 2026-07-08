from typing import Any, Literal

import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.io.data_sources import SqlDataSource
from pixeltable.utils import sql as sql_utils
from pixeltable.utils.sql import as_select


def export_sql(
    table_or_query: pxt.Table | pxt.Query,
    target_table_name: str,
    *,
    db_connect_str: str,
    target_schema_name: str | None = None,
    if_exists: Literal['error', 'replace', 'insert'] = 'error',
    if_not_exists: Literal['error', 'create'] = 'create',
) -> None:
    """
    Exports a query result or table to an RDBMS table.

    Args:
        table_or_query : Table or Query to export.
        target_table_name : Name of the target table.
        db_connect_str : Connection string to the target database.
        target_schema_name : Optional name of the target schema.
        if_exists : What to do if the target table already exists.

            - 'error': raise an error
            - 'replace': drop the existing table and create a new one
            - 'insert': insert new rows into the existing table
        if_not_exists : What to do if the target table does not exist.

            - 'error': raise an error
            - 'create': create the table from the source schema
    """
    # TODO:
    # - overrides for output schema
    # - include drop/create table in the data loading transaction
    # - merge flag

    query: pxt.Query
    if isinstance(table_or_query, pxt.Table):
        query = table_or_query.select()
    else:
        query = table_or_query

    engine = sql.create_engine(db_connect_str)
    try:
        target = sql_utils.resolve_table(
            engine=engine,
            table_name=target_table_name,
            schema_name=target_schema_name,
            source_schema=query.schema,
            if_exists=if_exists,
            if_not_exists=if_not_exists,
            error_prefix='export_sql()',
        )

        batch_size = 16 * 1024
        batch: list[dict] = []
        with engine.connect() as target_conn:
            for data_row in query.cursor():
                # we already preclude images so there isn't a need to convert media types here
                batch.append(dict(data_row))

                if len(batch) >= batch_size:
                    target_conn.execute(target.insert(), batch)
                    batch = []

            if len(batch) > 0:
                target_conn.execute(target.insert(), batch)
            target_conn.commit()

    except excs.ExprEvalError as e:
        query._raise_expr_eval_err(e)
    finally:
        engine.dispose()


def import_sql(
    selectable: sql.Selectable,
    conn: sql.Engine | sql.Connection,
    tbl_name: str,
    *,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
    comment: str | None = None,
    custom_metadata: Any = None,
    if_exists: Literal['error', 'append'] = 'error',
    on_error: Literal['abort', 'ignore'] = 'abort',
    send_connect_url: bool = False,
) -> pxt.Table:
    """Import a SQL source into a Pixeltable table.

    Rows are streamed from the source via a server-side cursor and inserted in batches.

    Args:
        selectable: A SQLAlchemy `Selectable` (a `Table`, a `select()` statement, or `text(...).columns(...)`)
            describing the source rows.
        conn: A SQLAlchemy `Engine` or `Connection` to execute `selectable` against. If a `Connection` is
            passed, it must remain open and untouched (no commits, rollbacks, or other statements) for the
            duration of the import; rows are streamed directly from the server-side cursor.
        tbl_name: Pixeltable path of the destination table.
        schema_overrides: Optional per-column overrides applied on top of the inferred schema. Keys are column
            names; values accept any Pixeltable type spec recognized by `pxt.create_table` (eg, `pxt.Image`,
            `pxt.Required[pxt.String]`).
        primary_key: An optional column name or list of column names to use as the primary key(s) of the
            table. Only applies when a new table is created; ignored when appending to an existing table.
        comment: An optional comment; its meaning is user-defined. Only applies when a new table is created;
            ignored when appending to an existing table.
        custom_metadata: Optional user-defined metadata to associate with the table. Must be a valid
            JSON-serializable object [str, int, float, bool, dict, list]. Only applies when a new table is
            created; ignored when appending to an existing table.
        if_exists: How to handle the destination table.

            - `'error'`: create the table; fail if it already exists.
            - `'append'`: append into the table if it already exists (verifying the source schema is
              compatible); otherwise create it.
        on_error: Determines the behavior if an error occurs while evaluating a computed column or detecting an
            invalid media file (such as a corrupt image) for one of the inserted rows.

            - If `on_error='abort'`, then an exception will be raised and the rows will not be inserted.
            - If `on_error='ignore'`, then execution will continue and the rows will be inserted. Any cells
              with errors will have a `None` value for that cell, with information about the error stored in the
              corresponding `tbl.col_name.errortype` and `tbl.col_name.errormsg` fields.
        send_connect_url: Determines where the query is run, and is only relevant when the destination table is hosted.
              It has no effect on a local table.

            - If `False`, the query is run locally and the result is sent to the server.
            - If `True`, the server connects to the source database itself (using the source connection URL, including
              credentials); this requires the server's host to be able to reach the database and is faster for large
              imports.

    Returns:
        The destination `Table`.
    """
    if if_exists not in ('error', 'append'):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            f"`if_exists` must be one of 'error', 'append'; got {if_exists!r}. To overwrite an existing table, "
            f"`pxt.drop_table(tbl_name)` first and then call `import_sql(..., if_exists='error')`.",
        )

    stmt = as_select(selectable)
    sa_cols = list(stmt.selected_columns)
    source_names: list[str] = []
    seen_cols: set[str] = set()
    schema_overrides = schema_overrides or {}
    for i, sa_col in enumerate(sa_cols):
        col_name = getattr(sa_col, 'name', None) or getattr(sa_col, 'key', None)
        if col_name is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'SQL source has an unnamed output column at position {i}; alias it via '
                "`expr.label('name')` so it can be matched to a Pixeltable column.",
            )
        if col_name in seen_cols:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'SQL source has duplicate output column {col_name!r}; output column names must be unique.',
            )
        seen_cols.add(col_name)
        source_names.append(col_name)

    schema: dict[str, Any] = {}
    for sa_col, col_name in zip(sa_cols, source_names, strict=True):
        # SQLAlchemy uses `None` for "unknown" (eg, on labeled expressions); treat that as nullable.
        nullable_attr = getattr(sa_col, 'nullable', True)
        nullable = True if nullable_attr is None else nullable_attr
        pxt_type: ts.ColumnType | None
        if col_name in schema_overrides:
            pxt_type = schema_overrides[col_name]
        else:
            pxt_type = ts.ColumnType.from_sa_type(sa_col.type, nullable=nullable)

        if pxt_type is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_TYPE,
                f'Cannot infer a Pixeltable type for SQL source column {col_name!r} (SQLAlchemy type '
                f'`{sa_col.type}`); provide an entry in `schema_overrides` to resolve it.',
            )

        schema[col_name] = pxt_type

    # Check that all schema_overrides keys are actually in the source schema; this helps catch typos
    unknown = set(schema_overrides) - set(schema)
    if len(unknown) > 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            f'`schema_overrides` references column(s) not in the SQL source: {", ".join(sorted(unknown))}',
        )

    def run(connection: sql.Connection) -> pxt.Table:
        sql_data_source = SqlDataSource(select_stmt=stmt, col_names=source_names, conn=connection)

        existing = pxt.get_table(tbl_name, if_not_exists='ignore')
        if if_exists == 'append' and existing is not None:
            if not isinstance(existing, (catalog.InsertableTable, catalog.InsertableTableProxy)):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'`import_sql` requires a base table; {tbl_name!r} is a {existing._display_name()}.',
                )
            _validate_append_compatibility(existing, tbl_name, schema)
            existing._insert_sql_source(sql_data_source, on_error=on_error, send_connect_url=send_connect_url)
            return existing

        tbl = pxt.create_table(
            tbl_name,
            schema=schema,
            primary_key=primary_key,
            comment=comment,
            custom_metadata=custom_metadata,
            if_exists='error',
        )
        try:
            tbl._insert_sql_source(sql_data_source, on_error=on_error, send_connect_url=send_connect_url)
        except BaseException:
            pxt.drop_table(tbl, if_not_exists='ignore')
            raise
        return tbl

    if isinstance(conn, sql.Engine):
        with conn.connect() as connection:
            return run(connection)
    return run(conn)


def _validate_append_compatibility(tbl: pxt.Table, tbl_name: str, schema: dict[str, Any]) -> None:
    """Verify the SQL source schema can append into an existing destination table."""
    column_md = tbl.get_metadata()['columns']
    existing_schema = tbl._get_schema()
    computed_col_names = {name for name, md in column_md.items() if md['is_computed']}
    # A column is required for insert if it is non-nullable and not computed.
    required_col_names = {
        name for name, col_type in existing_schema.items() if not col_type.nullable and name not in computed_col_names
    }

    for col_name, src_type_raw in schema.items():
        if col_name in computed_col_names:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'SQL source column {col_name!r} maps to computed column {col_name!r} in destination table '
                f'{tbl_name!r}; computed columns are populated automatically and cannot receive values.',
            )
        if col_name not in existing_schema:
            raise excs.NotFoundError(
                excs.ErrorCode.COLUMN_NOT_FOUND,
                f'SQL source column {col_name!r} does not match any column in destination table {tbl_name!r}.',
            )
        src_type = ts.ColumnType.normalize_type(src_type_raw, nullable_default=True, allow_builtin_types=False)
        dest_type = existing_schema[col_name]
        if not dest_type.is_supertype_of(src_type, ignore_nullable=True):
            raise excs.RequestError(
                excs.ErrorCode.TYPE_MISMATCH,
                f'SQL source column {col_name!r} has type `{src_type}`, which is incompatible with '
                f'destination column type `{dest_type}` in table {tbl_name!r}.',
            )

    missing_cols = required_col_names - set(schema)
    if missing_cols:
        raise excs.RequestError(
            excs.ErrorCode.MISSING_REQUIRED,
            f'Destination table {tbl_name!r} has required column(s) ({", ".join(sorted(missing_cols))}) '
            f'that are not provided by the SQL source.',
        )
