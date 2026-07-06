import sqlalchemy as sql

from pixeltable.metadata import register_converter


@register_converter(version=35)
def _(conn: sql.Connection) -> None:
    conn.execute(sql.text('ALTER TABLE tables ADD COLUMN lock_dummy int8'))
