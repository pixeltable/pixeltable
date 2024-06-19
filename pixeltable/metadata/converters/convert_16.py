import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=16)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(
        engine,
        table_md_updater=__update_table_md,
        column_md_updater=__update_column_md
    )


def __update_table_md(table_md: dict) -> None:
    # External stores are not migratable; just drop them
    del table_md['remotes']
    table_md['external_stores'] = {}


def __update_column_md(column_md: dict) -> None:
    column_md['proxy_base'] = None
