import logging
from typing import Any, Optional
from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md

_logger = logging.getLogger('pixeltable')


@register_converter(version=36)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, table_md_updater=__update_table_md, substitution_fn=__substitute_md)


def __update_table_md(table_md: dict, table_id: UUID) -> None:
    """Update the view metadata to add the sample_clause field if it is missing

    Args:
        table_md (dict): copy of the original table metadata. this gets updated in place.
        table_id (UUID): the table id

    """
    if table_md['view_md'] is None:
        return
    if 'sample_clause' not in table_md['view_md']:
        table_md['view_md']['sample_clause'] = None
        _logger.info(f'Updating view metadata for table: {table_id}')


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    if isinstance(v, dict) and (v.get('_classname') == 'DataFrame'):
        if 'sample_clause' not in v:
            v['sample_clause'] = None
        return k, v
    return None
