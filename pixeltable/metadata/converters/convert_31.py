import sqlalchemy as sql

from pixeltable.metadata import register_converter


@register_converter(version=31)
def _(conn: sql.Connection) -> None:
    # Add a column "lock_dummy: int8" to the dirs table in the store
    # This column is the target of an UPDATE operation to synchronize directory operations
    conn.execute(sql.text('ALTER TABLE dirs ADD COLUMN lock_dummy int8'))
