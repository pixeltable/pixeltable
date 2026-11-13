import sqlalchemy as sql

from pixeltable.metadata import register_converter


@register_converter(version=56)
def _(conn: sql.Connection) -> None:
    """
    Changes in version 57:
    - TableState.COPYING is the state of a table that a copy is still filling. No stored metadata changes.
    """
