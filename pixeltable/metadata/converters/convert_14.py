import sqlalchemy as sql

from pixeltable.metadata.schema import Table
from pixeltable.metadata import register_converter


def convert_14(engine: sql.engine.Engine) -> None:
    default_remotes = {'remotes': []}
    with engine.begin() as conn:
        conn.execute(sql.update(Table).where(Table.md['remotes'] == None).values(md=Table.md.concat(default_remotes)))


register_converter(14, convert_14)
