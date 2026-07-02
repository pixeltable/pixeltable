from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=22)
def _(conn: sql.Connection) -> None:
    convert_table_md(conn, substitution_fn=__substitute_md)


def __substitute_md(k: str | None, v: Any) -> tuple[str | None, Any] | None:
    if isinstance(v, dict) and '_classname' in v and v['_classname'] == 'DataFrame':
        v['from_clause'] = {'tbls': [v['tbl']], 'join_clauses': []}
        return k, v
    return None
