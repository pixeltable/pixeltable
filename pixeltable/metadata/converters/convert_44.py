from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=44)
def _(engine: sql.engine.Engine) -> None:
    """Adds a required parameter "precision" to EmbeddingIndex"""
    convert_table_md(engine, substitution_fn=_substitution_fn)


def _substitution_fn(key: str | None, value: Any) -> tuple[str | None, Any] | None:
    if not isinstance(value, dict):
        return None
    if ('class_fqn', 'pixeltable.index.embedding_index.EmbeddingIndex') not in value.items():
        return None
    if 'init_args' not in value:
        return None
    assert isinstance(value['init_args'], dict), value
    if 'precision' in value['init_args']:
        return None
    value['init_args']['precision'] = '32bit'
    return (key, value)
