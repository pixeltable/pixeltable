from typing import Any, Optional

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=26)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=__substitute_md)


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    import pixeltable.type_system as ts
    from pixeltable.exprs.literal import Literal

    if k == 'limit_val' and v is not None:
        assert isinstance(v, int)
        newv = Literal(v, ts.IntType(nullable=False)).as_dict()
        return k, newv

    return None
