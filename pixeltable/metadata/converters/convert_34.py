from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=34)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=__substitute_md)


def __substitute_md(k: str | None, v: Any) -> tuple[str | None, Any] | None:
    if isinstance(v, dict) and '_classname' in v and v['_classname'] == 'ColumnRef':
        # Add reference_tbl to ColumnRef; for historical metadata it is always equal to tbl
        assert 'reference_tbl' not in v
        v['reference_tbl'] = None
        return k, v

    return None
