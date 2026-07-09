from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import Table


@register_converter(version=14)
def _(conn: sql.Connection) -> None:
    default_remotes: dict[str, Any] = {'remotes': []}
    conn.execute(sql.update(Table).where(Table.md['remotes'] == None).values(md=Table.md.concat(default_remotes)))
