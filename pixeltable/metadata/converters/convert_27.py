import logging
from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md

_logger = logging.getLogger('pixeltable')


@register_converter(version=27)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, table_md_updater=__update_table_md)


def __update_table_md(table_md: dict, table_id: UUID) -> None:
    """Update the view metadata to add the include_base_columns boolean if it is missing

    Args:
        table_md (dict): copy of the original table metadata. this gets updated in place.
        table_id (UUID): the table id

    """
    if table_md['view_md'] is None:
        return
    if 'include_base_columns' not in table_md['view_md']:
        table_md['view_md']['include_base_columns'] = True
        _logger.info(f'Updating view metadata for table: {table_id}')
