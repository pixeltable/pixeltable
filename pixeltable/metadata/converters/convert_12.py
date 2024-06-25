import sqlalchemy as sql

from pixeltable.metadata import register_converter


@register_converter(version=12)
def _(engine: sql.engine.Engine) -> None:
    pass
