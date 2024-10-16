from typing import Any, Optional

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=21)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(
        engine,
        substitution_fn=__substitute_md
    )


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    if k == 'path' and v == 'pixeltable.index.btree.BtreeIndex.str_filter':
        return 'path', 'pixeltable.index.btree._str_filter'
    return None
