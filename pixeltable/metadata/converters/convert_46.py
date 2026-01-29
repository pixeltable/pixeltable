import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=45)
def _(engine: sql.engine.Engine) -> None:
    """Migration to add is_computed_column flag to ColumnMd."""
    convert_table_md(engine, column_md_updater=_update_column_md)


def _update_column_md(column_md: dict) -> None:
    """Update a single ColumnMd dict to use is_computed_column flag."""
    # Set is_computed_column=True for columns with value_expr (computed columns)
    if 'value_expr' in column_md and column_md['value_expr'] is not None:
        column_md['is_computed_column'] = True
    else:
        column_md['is_computed_column'] = False
