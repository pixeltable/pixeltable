import logging
from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md

_logger = logging.getLogger('pixeltable')


@register_converter(version=40)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, table_modifier=__table_modifier)


def __table_modifier(conn: sql.Connection, tbl_id: UUID, orig_table_md: dict, updated_table_md: dict) -> None:
    store_prefix = 'view' if orig_table_md['view_md'] is not None else 'tbl'
    store_name = f'{store_prefix}_{tbl_id.hex}'

    # Get the list of column names that need _cellmd columns
    _logger.info(f'Checking table {orig_table_md["name"]} ({store_name})')
    col_ids = find_target_columns(orig_table_md)
    if len(col_ids) == 0:
        _logger.info(f'No Array or Json columns found in table {orig_table_md["name"]}. Skipping migration.')
        return

    # Check which columns already exist in the table
    check_columns_sql = sql.text(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{store_name}'
    """)
    existing_columns = {row[0] for row in conn.execute(check_columns_sql)}

    # Filter out columns that already have _cellmd
    col_ids_to_add: list[int] = []
    for col_id in col_ids:
        cellmd_col = f'col_{col_id}_cellmd'
        if cellmd_col not in existing_columns:
            col_ids_to_add.append(col_id)
        else:
            _logger.info(f'Column {cellmd_col} already exists in table {orig_table_md["name"]}. Skipping.')

    if len(col_ids_to_add) == 0:
        _logger.info(f'All _cellmd columns already exist in table {orig_table_md["name"]}. Skipping migration.')
        return

    return add_cellmd_columns(conn, store_name, col_ids_to_add)


def find_target_columns(table_md: dict) -> list[int]:
    """Returns ids of stored array and json columns"""
    result: list[int] = []
    for col_id, col_md in table_md['column_md'].items():
        col_type = col_md['col_type']
        classname = col_type.get('_classname')
        if classname in ['ArrayType', 'JsonType'] and col_md.get('stored', False):
            result.append(col_id)
            _logger.info(f'Found {classname} column: {col_id}')
    return result


def add_cellmd_columns(conn: sql.Connection, store_name: str, col_ids: list[int]) -> None:
    try:
        # Add new columns
        add_column_str = ', '.join(f'ADD COLUMN col_{col_id}_cellmd JSONB DEFAULT NULL' for col_id in col_ids)
        add_column_sql = sql.text(f'ALTER TABLE {store_name} {add_column_str}')
        conn.execute(add_column_sql)
        _logger.info(f'Added columns to {store_name}: {", ".join(f"col_{col_id}_cellmd" for col_id in col_ids)}')
    except sql.exc.SQLAlchemyError as e:
        _logger.error(f'Migration for table {store_name} failed: {e}')
        raise
