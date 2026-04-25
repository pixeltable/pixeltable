from typing import Literal

import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.utils import sql as sql_utils


def export_sql(
    table_or_query: pxt.Table | pxt.Query,
    target_table_name: str,
    *,
    db_connect_str: str | None,
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
    target = sql_utils.resolve_table(
        engine=engine,
        table_name=target_table_name,
        schema_name=target_schema_name,
        source_schema=query.schema,
        if_exists=if_exists,
        if_not_exists=if_not_exists,
        error_prefix='',
    )

    batch_size = 16 * 1024
    try:
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
