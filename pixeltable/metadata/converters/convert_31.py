import sqlalchemy as sql

from pixeltable.metadata import register_converter


@register_converter(version=31)
def _(engine: sql.engine.Engine) -> None:
    # Add a column "lock_dummy: int8" to the dirs table in the store
    # This column is the target of an UPDATE operation to synchronize directory operations
    with engine.begin() as conn:
        conn.execute(sql.text('ALTER TABLE dirs ADD COLUMN lock_dummy int8'))
