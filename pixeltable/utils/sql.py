import datetime
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Literal

import sqlalchemy as sql
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import URL

import pixeltable.exceptions as excs

if TYPE_CHECKING:
    import pixeltable.type_system as ts


def log_stmt(logger: logging.Logger, stmt: sql.sql.ClauseElement) -> None:
    logger.debug(f'executing {stmt.compile(dialect=postgresql.dialect())}')


def log_explain(logger: logging.Logger, stmt: sql.sql.ClauseElement, conn: sql.engine.Connection) -> None:
    try:
        # don't set dialect=Env.get().engine.dialect: x % y turns into x %% y, which results in a syntax error
        stmt_str = str(stmt.compile(compile_kwargs={'literal_binds': True}))
        explain_result = conn.execute(sql.text(f'EXPLAIN {stmt_str}'))
        explain_str = '\n'.join(str(row) for row in explain_result)
        logger.debug(f'SqlScanNode explain:\n{explain_str}')
    except Exception:
        logger.warning('EXPLAIN failed')


def add_option_to_db_url(url: str | URL, option: str) -> URL:
    """Add a connection option to a database URL.

    Args:
        url: Database URL as string or SQLAlchemy URL object
        option: Option to add (e.g., '-c search_path=test_schema,public' or '-c timezone=UTC')

    Returns:
        Modified URL object with the option added to the query parameters
    """
    db_url = sql.make_url(url) if isinstance(url, str) else url

    # Get existing options and parse them
    # Query parameters can be strings or tuples (if multiple values exist)
    existing_options_raw = db_url.query.get('options', '') if db_url.query else ''
    option_parts = (
        list(existing_options_raw) if isinstance(existing_options_raw, tuple) else existing_options_raw.split()
    )
    option_parts.append(option)
    options_str = ' '.join(option_parts)

    # Create new URL with updated options
    return db_url.set(query={**(dict(db_url.query) if db_url.query else {}), 'options': options_str})


def _default_sa_type(col_type: 'ts.ColumnType') -> sql.types.TypeEngine:
    """Default mapping of ColumnType to SQLAlchemy type (matches sqlite, mysql)."""
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
    raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot export column of type {col_type}')


def _postgresql_sa_type(col_type: 'ts.ColumnType') -> sql.types.TypeEngine:
    """Type mapping for dialect 'postgresql'."""
    if col_type.is_json_type():
        return sql.dialects.postgresql.JSONB()
    return _default_sa_type(col_type)


def _snowflake_sa_type(col_type: 'ts.ColumnType') -> sql.types.TypeEngine:
    """Type mapping for dialect 'snowflake'."""
    from pixeltable.env import Env

    if col_type.is_json_type():
        Env.get().require_package(
            'snowflake.sqlalchemy',
            not_installed_msg='Exporting json data to Snowflake requires the snowflake-sqlalchemy package',
        )
        from snowflake.sqlalchemy import VARIANT  # type: ignore[import-untyped]

        return VARIANT()
    return _default_sa_type(col_type)


_DIALECT_TYPE: dict[str, Callable[['ts.ColumnType'], sql.types.TypeEngine]] = {
    'postgresql': _postgresql_sa_type,
    'snowflake': _snowflake_sa_type,
}


def get_sa_type(col_type: 'ts.ColumnType', engine: sql.Engine) -> sql.types.TypeEngine:
    """Resolve a Pixeltable ColumnType to an SQLAlchemy type appropriate for the engine's dialect."""
    return _DIALECT_TYPE.get(engine.dialect.name, _default_sa_type)(col_type)


def table_exists(engine: sql.Engine, table_name: str, schema_name: str | None = None) -> bool:
    """Check if a table exists in the database."""
    inspector = sql.inspect(engine)
    return table_name in inspector.get_table_names(schema=schema_name)


def _sample_literals() -> dict[Any, Any]:
    import pixeltable.type_system as ts

    return {
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


def _check_schema_compatible(
    table: sql.Table, source_schema: dict[str, 'ts.ColumnType'], engine: sql.Engine, error_prefix: str
) -> None:
    sample_literals = _sample_literals()
    with engine.connect() as conn:
        for col_name, source_type in source_schema.items():
            if col_name not in table.c:
                raise excs.NotFoundError(
                    excs.ErrorCode.COLUMN_NOT_FOUND, f'{error_prefix}: column {col_name!r} not in table {table.name!r}'
                )

            target_type = table.c[col_name].type
            # CAST(<literal> AS target_type)
            cast_expr = sql.cast(sample_literals[source_type._type], target_type).label(col_name)
            # 1 = 0: we only want to check whether the casts are legal, not run anything
            query = sql.select(cast_expr).where(sql.literal(1) == sql.literal(0))

            try:
                conn.execute(query)
            except Exception:
                raise excs.RequestError(
                    excs.ErrorCode.TYPE_MISMATCH,
                    f'{error_prefix}: in table {table.name!r}, column {col_name!r} of type {target_type} '
                    f'is not compatible with the source type ({source_type})',
                ) from None


def _create_table(
    engine: sql.Engine, table_name: str, schema_name: str | None, source_schema: dict[str, 'ts.ColumnType']
) -> sql.Table:
    columns = [sql.Column(col_name, get_sa_type(col_type, engine)) for col_name, col_type in source_schema.items()]
    metadata = sql.MetaData()
    table = sql.Table(table_name, metadata, *columns, schema=schema_name)
    table.create(engine, checkfirst=True)
    return table


def resolve_table(
    *,
    engine: sql.Engine,
    table_name: str,
    schema_name: str | None,
    source_schema: dict[str, 'ts.ColumnType'],
    if_exists: Literal['error', 'replace', 'insert'],
    if_not_exists: Literal['error', 'create'],
    error_prefix: str,
) -> sql.Table:
    """
    Resolve a target SQLAlchemy table for writes from a Pixeltable source schema.

    - exists, if_exists=error: raise AlreadyExistsError
    - exists, if_exists=replace: drop, then create from source_schema
    - exists, if_exists=insert: autoload and validate compatibility against source_schema
    - missing, if_not_exists=create: create from source_schema
    - missing, if_not_exists=error: raise NotFoundError

    error_prefix is prepended to raised messages so callers can attribute errors.
    """
    if table_exists(engine, table_name, schema_name):
        if if_exists == 'error':
            raise excs.AlreadyExistsError(
                excs.ErrorCode.PATH_ALREADY_EXISTS,
                f'{error_prefix}: table {table_name!r} already exists in:\n{engine.url}',
            )
        metadata = sql.MetaData()
        table = sql.Table(table_name, metadata, schema=schema_name, autoload_with=engine)
        if if_exists == 'replace':
            table.drop(engine)
            return _create_table(engine, table_name, schema_name, source_schema)
        _check_schema_compatible(table, source_schema, engine, error_prefix)
        return table

    if if_not_exists == 'error':
        raise excs.NotFoundError(
            excs.ErrorCode.PATH_NOT_FOUND, f'{error_prefix}: table {table_name!r} does not exist in:\n{engine.url}'
        )
    return _create_table(engine, table_name, schema_name, source_schema)
