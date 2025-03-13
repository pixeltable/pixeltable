from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import Table


@register_converter(version=14)
def _(engine: sql.engine.Engine) -> None:
    default_remotes: dict[str, Any] = {'remotes': []}
    with engine.begin() as conn:
        conn.execute(sql.update(Table).where(Table.md['remotes'] == None).values(md=Table.md.concat(default_remotes)))
