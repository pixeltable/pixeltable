import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_schema_version_md


@register_converter(version=21)
def _(engine: sql.engine.Engine) -> None:
    convert_table_schema_version_md(
        engine,
        table_schema_version_md_updater=__update_table_schema_version,
        schema_column_updater=__update_schema_column
    )


def __update_table_schema_version(table_schema_version_md: dict) -> None:
    table_schema_version_md['media_validation'] = 1  # MediaValidation.ON_WRITE


def __update_schema_column(schema_column: dict) -> None:
    schema_column['media_validation'] = None
