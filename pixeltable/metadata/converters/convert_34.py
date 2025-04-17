from typing import Any, Optional

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=34)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=__substitute_md)


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    if isinstance(v, dict) and '_classname' in v and v['_classname'] == 'ColumnRef':
        # Add reference_tbl to ColumnRef; for historical metadata it is always equal to tbl
        assert 'reference_tbl_id' not in v
        assert 'reference_tbl_version' not in v
        v['reference_tbl_id'] = v['tbl_id']
        v['reference_tbl_version'] = v['tbl_version']
        return k, v

    return None
