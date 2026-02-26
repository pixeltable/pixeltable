import sqlalchemy as sql

from pixeltable.metadata import register_converter


@register_converter(version=45)
def _(engine: sql.engine.Engine) -> None:
    """No-op migration (was: add is_computed_column; now using default_value_expr instead)."""
    pass
