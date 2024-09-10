from typing import Any, Optional
import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=19)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(
        engine,
        substitution_fn=__substitute_md
    )


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    if isinstance(v, dict) and '_classname' in v and v['_classname'] == 'InlineArray':
        # Assume an `InlineArray` from schema version <= 19 should actually be an
        # `InlineList`. This was formerly ambiguous in metadata,
        # but assuming it's an `InlineList` maps to the most common deployed use cases.
        updated_v = {
            'elements': v['elements'],
            '_classname': 'InlineList'
        }
        if 'components' in v:
            updated_v['components'] = v['components']
        return k, updated_v
    return None
