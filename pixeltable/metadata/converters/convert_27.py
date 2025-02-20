from typing import Any, Optional

import sqlalchemy as sql

from pixeltable import exprs
from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=27)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=__substitute_md)


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    # Defaults are now stored as literals in signatures
    if k == 'signature':
        assert isinstance(v, dict)
        newv = v.copy()
        has_default = 'has_default' in v or (v.get('default') is not None)
        if 'has_default' in newv:
            del newv['has_default']
        literal = exprs.Expr.from_object(v['default']) if has_default else None
        assert literal is None or isinstance(literal, exprs.Literal)
        newv['default'] = None if literal is None else literal.as_dict()
        return k, newv

    if k == 'fn':
        assert False, v

    return None
