from typing import Any, Optional

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=21)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(
        engine, table_md_updater=__update_tbl_md, column_md_updater=__update_column_md
    )


def __update_tbl_md(tbl_md: dict) -> None:
    tbl_md['media_validation'] = 1  # catalog.MediaValidation.ON_WRITE


def __update_column_md(column_md: dict) -> None:
    column_md['media_validation'] = None