from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=33)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, table_md_updater=__update_table_md)


def __update_table_md(table_md: dict, table_id: UUID) -> None:
    """Set default value of 'is_pk' field in column metadata to False"""
    for col_md in table_md['column_md'].values():
        col_md['is_pk'] = False if col_md['is_pk'] is None else col_md['is_pk']
