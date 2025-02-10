from typing import Any, Optional

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=26)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=__substitute_md)


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    from pixeltable.exprs.literal import Literal
    import pixeltable.type_system as ts
    import pixeltable.exprs as exprs
    if k == 'limit_val' and v is not None:
        if isinstance(v, int):
            newv = exprs.Literal(v, ts.IntType(nullable=False)).as_dict()
            print("*************************************", newv)
            return k, newv

    return None
