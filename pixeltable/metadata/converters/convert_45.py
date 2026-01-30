from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=45)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=_substitution_fn)


def _substitution_fn(key: str | None, value: Any) -> tuple[str | None, Any] | None:
    if not isinstance(value, dict):
        return None
    if 'iterator_class_fqn' in value:
        assert 'iterator_args' in value
        iterator_class_fqn = value['iterator_class_fqn']
        iterator_args = value['iterator_args']
        if iterator_class_fqn is None:
            assert iterator_args is None
            del value['iterator_class_fqn']
            del value['iterator_args']
            value['iterator_call'] = None
        else:
            assert False, value

    return key, value
