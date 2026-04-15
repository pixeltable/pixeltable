from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=50)
def _(engine: sql.engine.Engine) -> None:
    """Backfill the 'is_versioned' field in the pre-existing Table metadata"""
    convert_table_md(engine, table_md_updater=_update_table_md)


def _update_table_md(table_md: dict, table_id: UUID) -> None:
    assert 'is_versioned' not in table_md
    table_md['is_versioned'] = True
