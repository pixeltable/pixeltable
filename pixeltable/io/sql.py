"""
To add support for a new target database:
- check if the generic type mapping of _get_sa_type() needs to be overridden
- if so, add a new function _get_{dialect}_type()
- add an entry to GET_DIALECT_TYPE
"""

import datetime
import uuid
from typing import Any, Callable, Literal

import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.catalog import Catalog
from pixeltable.env import Env


def _sa_type(col_type: ts.ColumnType) -> sql.types.TypeEngine:
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


def _postgresql_type(col_type: ts.ColumnType) -> sql.types.TypeEngine:
    """Type mapping for dialect 'postgresql'"""
    if col_type.is_json_type():
        return sql.dialects.postgresql.JSONB()
    return _sa_type(col_type)


def _snowflake_type(col_type: ts.ColumnType) -> sql.types.TypeEngine:
    """Type mapping for dialect 'snowflake'"""
    if col_type.is_json_type():
        Env.get().require_package(
            'snowflake.sqlalchemy',
            not_installed_msg='Exporting json data to Snowflake requires the snowflake-sqlalchemy package',
        )
        from snowflake.sqlalchemy import VARIANT  # type: ignore[import-untyped]

        return VARIANT()
    else:
        return _sa_type(col_type)


DIALECT_TYPE: dict[str, Callable[[ts.ColumnType], sql.types.TypeEngine]] = {
    'postgresql': _postgresql_type,
    'snowflake': _snowflake_type,
}


def export_sql(
    table_or_query: pxt.Table | pxt.Query,
    target_table_name: str,
    *,
    db_connect_str: str | None,
    target_schema_name: str | None = None,
    if_exists: Literal['error', 'replace', 'insert'] = 'error',
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

    metadata = sql.MetaData()
    target: sql.Table | None = None
    if _table_exists(engine, target_table_name, target_schema_name):
        if if_exists == 'error':
            raise pxt.Error(f'Table {target_table_name!r} already exists in:\n{db_connect_str}')
        target = sql.Table(target_table_name, metadata, schema=target_schema_name, autoload_with=engine)
        if if_exists == 'replace':
            # drop existing table first
            target.drop(engine)
            target = None
        else:
            _check_schema_compatible(target, query.schema, engine)

    if target is None:
        # create table
        dialect = engine.dialect.name
        get_type = DIALECT_TYPE.get(dialect, _sa_type)
        target_schema = {col_name: get_type(col_type) for col_name, col_type in query.schema.items()}
        columns = [sql.Column(col_name, col_type) for col_name, col_type in target_schema.items()]
        metadata = sql.MetaData()
        target = sql.Table(target_table_name, metadata, *columns, schema=target_schema_name)
        target.create(engine, checkfirst=True)

    batch_size = 16 * 1024
    try:
        batch: list[dict] = []
        with Catalog.get().begin_xact(for_write=False), engine.connect() as target_conn:
            for data_row in query._exec():
                row_dict: dict[str, Any] = {}
                for col_name, e in zip(query.schema.keys(), query._select_list_exprs):
                    row_dict[col_name] = data_row[e.slot_idx]
                batch.append(row_dict)

                if len(batch) >= batch_size:
                    target_conn.execute(target.insert(), batch)
                    batch = []

            if len(batch) > 0:
                target_conn.execute(target.insert(), batch)
            target_conn.commit()

    except excs.ExprEvalError as e:
        query._raise_expr_eval_err(e)


def _table_exists(engine: sql.Engine, table_name: str, schema_name: str | None = None) -> bool:
    """Check if a table exists in the database."""
    inspector = sql.inspect(engine)
    return table_name in inspector.get_table_names(schema=schema_name)


_SAMPLE_LITERALS: dict[ts.ColumnType.Type, Any] = {
    ts.ColumnType.Type.STRING: 'test',
    ts.ColumnType.Type.INT: 1,
    ts.ColumnType.Type.FLOAT: 1.0,
    ts.ColumnType.Type.BOOL: True,
    # don't reference Env.default_time_zone here, Env may not be initialized yet
    ts.ColumnType.Type.TIMESTAMP: datetime.datetime.now(tz=datetime.timezone.utc),
    ts.ColumnType.Type.DATE: datetime.date.today(),
    ts.ColumnType.Type.UUID: uuid.uuid4(),
    ts.ColumnType.Type.BINARY: b'test',
    ts.ColumnType.Type.JSON: {'a': 1, 'b': [2, 3], 'c': {'d': 4}},
}


def _check_schema_compatible(tbl: sql.Table, source_schema: dict[str, ts.ColumnType], engine: sql.Engine) -> None:
    for col_name, source_type in source_schema.items():
        if col_name not in tbl.c:
            raise pxt.Error(f'Column {col_name!r} not in table {tbl.name!r}')

        target_type = tbl.c[col_name].type
        # CAST(<literal> AS target_type)
        cast_expr = sql.cast(_SAMPLE_LITERALS[source_type._type], target_type).label(col_name)
        # 1 = 0: we only want to check whether the casts are legal, not run anything
        query = sql.select(cast_expr).where(sql.literal(1) == sql.literal(0))

        with engine.connect() as conn:
            try:
                conn.execute(query)
            except Exception:
                raise pxt.Error(
                    f'Table {tbl.name!r}: column {col_name!r} of type {target_type} is not compatible with '
                    f'the source type ({source_type})'
                ) from None
