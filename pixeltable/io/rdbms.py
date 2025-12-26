from typing import Any, Literal

import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.catalog import Catalog


def export_rdbms(
    table_or_query: pxt.Table | pxt.Query,
    table_name: str,
    *,
    connection_string: str | None,
    schema_name: str | None = None,
    if_exists: Literal['error', 'overwrite', 'append'] = 'append',
) -> None:
    """
    Exports a query result or table to a RDBMS table.

    Args:
        table_or_query : Table or Query to export.
        table_name : Name of the target table.
        connection_string : Connection string to the target database.
        schema_name : Optional name of the target schema.
        if_exists : What to do if the target table already exists.
    """

    query: pxt.Query
    if isinstance(table_or_query, pxt.Table):
        query = table_or_query.select()
    else:
        query = table_or_query
    source_schema = {col_name: col_type.to_sa_type() for col_name, col_type in query.schema.items()}

    engine = sql.create_engine(connection_string)
    metadata = sql.MetaData()
    target: sql.Table | None = None
    if _table_exists(engine, table_name, schema_name):
        if if_exists == 'error':
            raise pxt.Error(f'Table {table_name!r} already exists in {connection_string!r}')
        target = sql.Table(table_name, metadata, schema=schema_name, autoload_with=engine)
        if if_exists == 'overwrite':
            # drop existing table first
            target.drop(engine)
            target = None
        else:
            _check_schema_compatible(target, source_schema, engine)

    if target is None:
        # create table
        columns = [sql.Column(col_name, col_type) for col_name, col_type in source_schema.items()]
        target = sql.Table(table_name, metadata, *columns, schema=schema_name)
        target.create(engine, checkfirst=True)

    batch_size = 1024
    try:
        batch: list[dict] = []
        with Catalog.get().begin_xact(for_write=False):
            for data_row in query._exec():
                row_dict: dict[str, Any] = {}
                for (col_name, _), e in zip(query.schema.items(), query._select_list_exprs):
                    row_dict[col_name] = data_row[e.slot_idx]
                batch.append(row_dict)

                if len(batch) >= batch_size:
                    with engine.connect() as target_conn:
                        target_conn.execute(target.insert(), batch)
                        target_conn.commit()
                    batch = []

            if len(batch) > 0:
                with engine.connect() as target_conn:
                    target_conn.execute(target.insert(), batch)
                    target_conn.commit()

    except excs.ExprEvalError as e:
        query._raise_expr_eval_err(e)


def _table_exists(engine: sql.Engine, table_name: str, schema_name: str | None = None) -> bool:
    """Check if a table exists in the database."""
    inspector = sql.inspect(engine)
    return table_name in inspector.get_table_names(schema=schema_name)


def _check_schema_compatible(
    tbl: sql.Table, source_schema: dict[str, sql.types.TypeEngine], engine: sql.Engine
) -> None:
    try:
        cast_exprs: list[sql.sql.ColumnElement] = []
        for col_name, source_type in source_schema.items():
            if col_name not in tbl.c:
                raise pxt.Error(f'Column {col_name!r} not in table {tbl.name!r}')

            target_type = tbl.c[col_name].type
            # CAST(CAST(NULL AS source_type) AS target_type)
            expr = sql.cast(sql.cast(sql.literal(None), source_type), target_type).label(col_name)
            cast_exprs.append(expr)
        # 1 = 0: we only want to check whether the casts are legal, not run anything
        query = sql.select(*cast_exprs).where(sql.literal(1) == sql.literal(0))

        with engine.connect() as conn:
            conn.execute(query)
    except Exception as e:
        raise pxt.Error(f'Table {tbl.name!r} is not compatible with the source: {e}')
