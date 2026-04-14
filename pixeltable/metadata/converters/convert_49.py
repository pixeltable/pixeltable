import sqlalchemy as sql

from pixeltable.metadata import register_converter


@register_converter(version=49)
def _(engine: sql.engine.Engine) -> None:
    """No-op: return_scalar backward compat handled by QueryTemplateFunction._from_dict."""
    pass
