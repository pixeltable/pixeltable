import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import Dir, Table, TableSchemaVersion, TableVersion


@register_converter(version=28)
def _(engine: sql.engine.Engine) -> None:
    with engine.begin() as conn:
        conn.execute(sql.update(Dir).values(md=Dir.md.concat({'user': None, 'additional_md': {}})))
        conn.execute(sql.update(Table).values(md=Table.md.concat({'user': None, 'additional_md': {}})))
        conn.execute(sql.update(TableVersion).values(md=TableVersion.md.concat({'additional_md': {}})))
        conn.execute(sql.update(TableSchemaVersion).values(md=TableSchemaVersion.md.concat({'additional_md': {}})))
