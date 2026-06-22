from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=32)
def _(conn: sql.Connection) -> None:
    convert_table_md(conn, table_md_updater=__update_table_md)


def __update_table_md(table_md: dict, table_id: UUID) -> None:
    table_md['is_replica'] = False
