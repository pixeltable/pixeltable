import logging
from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md

_logger = logging.getLogger('pixeltable')


@register_converter(version=23)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, table_md_updater=__update_table_md)


def __update_table_md(table_md: dict, table_id: UUID) -> None:
    """update the index metadata to add indexed_col_tbl_id column if it is missing

    Args:
        table_md (dict): copy of the original table metadata. this gets updated in place.
        table_id (UUID): the table id

    """
    if len(table_md['index_md']) == 0:
        return
    for idx_md in table_md['index_md'].values():
        if 'indexed_col_tbl_id' not in idx_md:
            # index metadata is missing indexed_col_tbl_id
            # assume that the indexed column is in the same table
            # and update the index metadata.
            _logger.info(f'Updating index metadata for table: {table_id} index: {idx_md["id"]}')
            idx_md['indexed_col_tbl_id'] = str(table_id)
