import sqlalchemy as sql

from pixeltable.metadata.schema import Table, TableSchemaVersion
from pixeltable.metadata import register_converter


def convert_12(engine: sql.engine.Engine) -> None:
    default_remotes = {'remotes': []}
    with engine.begin() as conn:
        conn.execute(sql.update(Table).values(md=Table.md.concat(default_remotes)))


register_converter(12, convert_12)
