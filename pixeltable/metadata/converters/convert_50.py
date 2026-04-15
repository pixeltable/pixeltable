import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import TableSchemaVersion


@register_converter(version=50)
def _(engine: sql.engine.Engine) -> None:
    with engine.begin() as conn:
        conn.execute(sql.update(TableSchemaVersion).values(md=TableSchemaVersion.md - 'num_retained_versions'))
