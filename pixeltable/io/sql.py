"""
To add support for a new target database:
- check if the generic type mapping of _get_sa_type() needs to be overridden
- if so, add a new function _get_{dialect}_type()
- add an entry to GET_DIALECT_TYPE
"""
from typing import Any, Callable, Literal

import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.catalog import Catalog


def _get_sa_type(col_type: ts.ColumnType) -> sql.types.TypeEngine:
    """
    Default mapping of ColumnType to SQLAlchemy type.

    This matches the following dialects:
    - sqlite
    - mysql
    """
    if col_type.is_int_type():
        return sql.Integer()
    elif col_type.is_string_type():
        return sql.String()
    elif col_type.is_float_type():
        return sql.Float()
    elif col_type.is_bool_type():
        return sql.Boolean()
    elif col_type.is_timestamp_type():
        return sql.TIMESTAMP()
    elif col_type.is_date_type():
        return sql.Date()
    elif col_type.is_uuid_type():
        return sql.UUID()
    elif col_type.is_binary_type():
        return sql.LargeBinary()
    elif col_type.is_json_type():
        return sql.JSON()
    raise pxt.Error(f'Cannot export column of type {col_type}')


def _get_postgresql_type(col_type: ts.ColumnType) -> sql.types.TypeEngine:
    """Type mapping for dialect 'postgresql'"""
    if col_type.is_json_type():
        return sql.dialects.postgresql.JSONB()
    return _get_sa_type(col_type)


def _get_snowflake_type(col_type: ts.ColumnType) -> sql.types.TypeEngine:
    """Type mapping for dialect 'snowflake'"""
    if col_type.is_json_type():
        try:
            from snowflake.sqlalchemy import VARIANT  # type: ignore[import-not-found]

            return VARIANT()
        except ImportError:
            raise pxt.Error(
                'In order to export json data to Snowflake, please install the snowflake-sqlalchemy package:\n'
                'pip install snowflake-sqlalchemy'
            ) from None
    return _get_sa_type(col_type)


GET_DIALECT_TYPE: dict[str, Callable[[ts.ColumnType], sql.types.TypeEngine]] = {
    'postgresql': _get_postgresql_type,
    'snowflake': _get_snowflake_type,
}


def export_sql(
    table_or_query: pxt.Table | pxt.Query,
    table_name: str,
    *,
    connection_string: str | None,
    schema_name: str | None = None,
    if_exists: Literal['error', 'replace', 'append'] = 'append',
) -> None:
    """
    Exports a query result or table to a RDBMS table.

    Args:
        table_or_query : Table or Query to export.
        table_name : Name of the target table.
        connection_string : Connection string to the target database.
        schema_name : Optional name of the target schema.
        if_exists : What to do if the target table already exists.

    TODO:
    - overrides for output schema
    """

    query: pxt.Query
    if isinstance(table_or_query, pxt.Table):
        query = table_or_query.select()
    else:
        query = table_or_query

    engine = sql.create_engine(connection_string)
    dialect = engine.dialect.name
    get_type = GET_DIALECT_TYPE.get(dialect, _get_sa_type)
    source_schema = {col_name: get_type(col_type) for col_name, col_type in query.schema.items()}

    metadata = sql.MetaData()
    target: sql.Table | None = None
    if _table_exists(engine, table_name, schema_name):
        if if_exists == 'error':
            raise pxt.Error(f'Table {table_name!r} already exists in {connection_string!r}')
        target = sql.Table(table_name, metadata, schema=schema_name, autoload_with=engine)
        if if_exists == 'replace':
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

    batch_size = 16 * 1024
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
        raise pxt.Error(f'Table {tbl.name!r} is not compatible with the source: {e}') from None
