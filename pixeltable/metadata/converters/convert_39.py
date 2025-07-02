import logging
from typing import Optional
from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md

_logger = logging.getLogger('pixeltable')


@register_converter(version=39)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, table_modifier=__table_modifier)


def __table_modifier(conn: sql.Connection, tbl_id: UUID, orig_table_md: dict, updated_table_md: dict) -> None:
    store_prefix = 'view' if orig_table_md['view_md'] is not None else 'tbl'
    store_name = f'{store_prefix}_{tbl_id.hex}'

    # Get the list of column names that need to be migrated
    col_names = find_error_columns(conn=conn, store_name=store_name)
    if len(col_names) == 0:
        _logger.info(f'No error columns found in table {store_name}. Skipping migration.')
        return

    # Check if the table exists, outside of the metadata we were given
    # There seem to be cases where the metadata is present in the catalog,
    # but the table itself is not in the database.
    check_table_sql = sql.text(f"""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = '{store_name}'
        )
    """)
    table_exists = conn.execute(check_table_sql).scalar()
    if not table_exists:
        _logger.warning(f'Table {store_name} does not exist. Skipping migration.')
        return

    return migrate_error_to_cellmd_columns(conn, store_name, col_names)


def find_error_columns(conn: sql.Connection, store_name: str) -> list[str]:
    """
    Return and errormsg or errortype columns in the given table

    Args:
        conn: SQLAlchemy connection
        store_name: Name of the table to check

    Returns:
        List of column name roots (root_errormsg, root_errortype)
    """
    check_columns_sql = sql.text(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{store_name}'
    """)
    found_columns = [
        row[0]
        for row in conn.execute(check_columns_sql)
        if row[0].endswith('_errormsg') or row[0].endswith('_errortype')
    ]
    column_roots = {s.removesuffix('_errormsg').removesuffix('_errortype') for s in found_columns}
    return [*column_roots]


def migrate_error_to_cellmd_columns(
    conn: sql.Connection, store_name: str, col_names: list[str], backup_table: Optional[str] = None
) -> None:
    """
    Safe version with error handling and optional backup.

    Args:
        engine: SQLAlchemy engine
        store_name: Name of the table to modify
        col_names: List of column name prefixes
        backup_table: Optional name for backup table

    Usage:
        migrate_error_to_cellmd_columns(engine, 'my_table', ['columnname'], 'my_table_backup')
    """

    try:
        # Optional: Create backup
        if backup_table:
            backup_sql = sql.text(f"""
                CREATE TABLE {backup_table} AS SELECT * FROM {store_name}
            """)
            conn.execute(backup_sql)
            _logger.info(f'Backup created: {backup_table}')

        # Step 1: Add new columns
        add_column_str = ', '.join(f'ADD COLUMN {col}_cellmd JSONB DEFAULT NULL' for col in col_names)
        add_column_sql = sql.text(f'ALTER TABLE {store_name} {add_column_str}')
        conn.execute(add_column_sql)
        _logger.info(f'Added columns: {", ".join(f"{col}_cellmd" for col in col_names)}')

        # Step 2: Populate new columns
        set_column_str = ', '.join(
            [
                f'{col}_cellmd = CASE WHEN {col}_errormsg IS NULL OR {col}_errortype IS NULL '
                f"THEN NULL ELSE jsonb_build_object('errormsg', {col}_errormsg, 'errortype', {col}_errortype) END"
                for col in col_names
            ]
        )
        populate_sql = sql.text(f'UPDATE {store_name} SET {set_column_str}')
        result = conn.execute(populate_sql)
        _logger.info(f'Updated {result.rowcount} rows')

        # Step 3: Drop old columns
        drop_columns_str = ', '.join(
            [f'DROP COLUMN IF EXISTS {col}_errormsg, DROP COLUMN IF EXISTS {col}_errortype' for col in col_names]
        )
        drop_columns_sql = sql.text(f'ALTER TABLE {store_name} {drop_columns_str}')
        conn.execute(drop_columns_sql)
        _logger.info(f'Dropped columns: {", ".join(f"{col}_errormsg, {col}_errortype" for col in col_names)}')
        _logger.info(f'Migration completed successfully for table: {store_name}')

    except sql.exc.SQLAlchemyError as e:
        _logger.error(f'Migration for table {store_name} failed: {e}')
        raise
