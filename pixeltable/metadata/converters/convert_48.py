import logging
from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md
from pixeltable.metadata.schema import Table

_logger = logging.getLogger('pixeltable')

MAX_VERSION = 9223372036854775807  # 2^63 - 1
MAX_STRING_LEN = 256  # Must match BtreeIndex.MAX_STRING_LEN


@register_converter(version=48)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, table_modifier=_table_modifier)


def _table_modifier(conn: sql.Connection, tbl_id: UUID, orig_table_md: dict, updated_table_md: dict) -> None:
    """Attempt to create primary key indexes for tables that have PK metadata.

    Uses a savepoint so that failure (e.g. due to duplicate values) does not abort the
    enclosing transaction. On failure the PK metadata is removed from the table.
    """
    primary_index_md = updated_table_md.get('primary_index_md')
    if primary_index_md is None:
        return

    indexed_col_ids = primary_index_md.get('indexed_col_ids', [])
    if not indexed_col_ids:
        return

    store_prefix = 'view' if updated_table_md.get('view_md') is not None else 'tbl'
    store_name = f'{store_prefix}_{tbl_id.hex}'
    idx_name = f'pk_idx_{tbl_id.hex}'

    result = conn.execute(
        sql.text('SELECT 1 FROM pg_indexes WHERE tablename = :tbl AND indexname = :idx'),
        {'tbl': store_name, 'idx': idx_name},
    )
    if result.fetchone() is not None:
        _logger.info(f'Primary key index {idx_name} already exists on {store_name}, skipping')
        return

    idx_exprs: list[str] = []
    for col_id in indexed_col_ids:
        col_md = updated_table_md['column_md'].get(str(col_id))
        if col_md is None:
            _logger.warning(f'Column {col_id} not found in table metadata for {store_name}, skipping PK index')
            return
        if col_md.get('schema_version_drop') is not None:
            _logger.warning(f'PK column {col_id} was dropped in {store_name}, skipping PK index')
            return
        col_name = f'col_{col_id}'
        if col_md['col_type'].get('_classname') == 'StringType':
            idx_exprs.append(f'left({col_name}, {MAX_STRING_LEN})')
        else:
            idx_exprs.append(col_name)

    create_idx_sql = (
        f'CREATE UNIQUE INDEX {idx_name} ON {store_name} '
        f'USING btree ({", ".join(idx_exprs)}) '
        f'WHERE v_max = {MAX_VERSION}'
    )

    conn.execute(sql.text('SAVEPOINT pk_index_attempt'))
    try:
        conn.execute(sql.text(create_idx_sql))
        conn.execute(sql.text('RELEASE SAVEPOINT pk_index_attempt'))
        _logger.info(f'Created primary key index {idx_name} on {store_name}')
    except (sql.exc.IntegrityError, sql.exc.InternalError) as e:
        _logger.warning(f'Failed to create PK index on {store_name}: {e}. Removing PK metadata.')
        conn.execute(sql.text('ROLLBACK TO SAVEPOINT pk_index_attempt'))

        for col_id in indexed_col_ids:
            col_key = str(col_id)
            if col_key in updated_table_md['column_md']:
                updated_table_md['column_md'][col_key]['is_pk'] = False
        updated_table_md['primary_index_md'] = None
        conn.execute(sql.update(Table).where(Table.id == tbl_id).values(md=updated_table_md))
