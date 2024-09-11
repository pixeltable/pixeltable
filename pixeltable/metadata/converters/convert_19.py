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
    if isinstance(v, dict) and '_classname' in v:
        if v['_classname'] == 'InlineArray':
            # Assume an `InlineArray` from schema version <= 19 should actually be an
            # `InlineList`. This was formerly ambiguous in metadata,
            # but assuming it's an `InlineList` maps to the most common deployed use cases.
            updated_elements = []
            for idx, val in v['elements']:
                # Replace any -1's that show up as indices with Nones
                # (this corrects for an older legacy inconsistency)
                updated_elements.append((None if idx == -1 else idx, val))
            updated_v = v.copy()
            updated_v['elements'] = updated_elements
            if 'is_json' in updated_v:
                if updated_v['is_json']:
                    # If the InlineArray is JSON, convert it to a list
                    updated_v['_classname'] = 'InlineList'
                del updated_v['is_json']
            return k, updated_v
        if v['_classname'] == 'InlineDict':
            updated_dict_items = []
            for key, idx, val in v['dict_items']:
                # Replace any -1's that show up as indices with Nones
                updated_dict_items.append((key, None if idx == -1 else idx, val))
            updated_v = v.copy()
            updated_v['dict_items'] = updated_dict_items
            return k, updated_v
    return None
