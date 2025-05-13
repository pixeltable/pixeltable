import sqlalchemy as sql

from pixeltable.metadata import register_converter


@register_converter(version=35)
def _(engine: sql.engine.Engine) -> None:
    with engine.begin() as conn:
        conn.execute(sql.text('ALTER TABLE tables ADD COLUMN lock_dummy int8'))
