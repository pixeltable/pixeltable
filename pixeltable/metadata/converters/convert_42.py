import sqlalchemy as sql

from pixeltable.metadata import register_converter


@register_converter(version=42)
def _(conn: sql.Connection) -> None:
    conn.execute(sql.text('ALTER TABLE tables ALTER COLUMN dir_id DROP NOT NULL'))
