from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=25)
def _(conn: sql.Connection) -> None:
    convert_table_md(conn, substitution_fn=__substitute_md)


def __substitute_md(k: str | None, v: Any) -> tuple[str | None, Any] | None:
    if k == 'path' and (
        v in ('pixeltable.functions.huggingface.clip_text', 'pixeltable.functions.huggingface.clip_image')
    ):
        return 'path', 'pixeltable.functions.huggingface.clip'
    return None
