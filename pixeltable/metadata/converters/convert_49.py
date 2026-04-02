import logging
from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md

_logger = logging.getLogger('pixeltable')


@register_converter(version=49)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, table_modifier=_table_modifier)


def _table_modifier(conn: sql.Connection, tbl_id: UUID, orig_table_md: dict, updated_table_md: dict) -> None:
    """Create primary key indexes for tables that have is_pk columns but no index yet.

    Old tables only have is_pk on individual columns in column_md; primary_index_md didn't
    exist before this migration. This converter builds the primary_index_md and creates the
    corresponding PostgreSQL unique index. On failure (e.g. duplicates), the PK metadata is
    removed.
    """
    from pixeltable.index.btree import BtreeIndex
    from pixeltable.metadata.schema import Table

    # Find columns marked as PK in the old metadata
    pk_col_ids: list[int] = []
    for col_id_str, col_md in updated_table_md['column_md'].items():
        if col_md.get('is_pk') is True and col_md.get('schema_version_drop') is None:
            pk_col_ids.append(int(col_id_str))

    if not pk_col_ids:
        return

    if updated_table_md.get('view_md') is not None:
        return

    store_name = f'tbl_{tbl_id.hex}'
    idx_name = f'pk_idx_{tbl_id.hex}'

    result = conn.execute(
        sql.text('SELECT 1 FROM pg_indexes WHERE tablename = :tbl AND indexname = :idx'),
        {'tbl': store_name, 'idx': idx_name},
    )
    if result.fetchone() is not None:
        _logger.info(f'Primary key index {idx_name} already exists on {store_name}, skipping')
        return

    idx_exprs: list[str] = []
    for col_id in pk_col_ids:
        col_md = updated_table_md['column_md'][str(col_id)]
        col_name = f'col_{col_id}'
        if col_md['col_type'].get('_classname') == 'StringType':
            idx_exprs.append(f'left({col_name}, {BtreeIndex.MAX_STRING_LEN})')
        else:
            idx_exprs.append(col_name)

    create_idx_sql = (
        f'CREATE UNIQUE INDEX {idx_name} ON {store_name} '
        f'USING btree ({", ".join(idx_exprs)}) '
        f'WHERE v_max = {Table.MAX_VERSION}'
    )

    conn.execute(sql.text('SAVEPOINT pk_index_attempt'))
    try:
        conn.execute(sql.text(create_idx_sql))
        conn.execute(sql.text('RELEASE SAVEPOINT pk_index_attempt'))
        _logger.info(f'Created primary key index {idx_name} on {store_name}')
    except (sql.exc.IntegrityError, sql.exc.InternalError) as e:
        _logger.info(f'Failed to create PK index on {store_name}: {e}. Removing PK metadata.')
        conn.execute(sql.text('ROLLBACK TO SAVEPOINT pk_index_attempt'))
        conn.execute(sql.text('RELEASE SAVEPOINT pk_index_attempt'))
        for col_id in pk_col_ids:
            updated_table_md['column_md'][str(col_id)]['is_pk'] = False

    conn.execute(sql.update(Table).where(Table.id == tbl_id).values(md=updated_table_md))
