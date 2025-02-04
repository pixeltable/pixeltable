from typing import Any, Optional

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=20)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=__substitute_md)


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    if isinstance(v, dict) and '_classname' in v:
        # The way InlineArray is represented changed in v20. Previously, literal values were stored
        # directly in the Inline expr; now we store them in Literal sub-exprs. This converter
        # constructs new Literal exprs for the literal values in InlineArray, interleaving them
        # with non-literal exprs into the correct sequence.
        if v['_classname'] == 'InlineArray':
            components = v.get('components')  # Might be None, but that's ok
            updated_components = []
            for idx, val in v['elements']:
                # idx >= 0, then this is a non-literal sub-expr. Otherwise, idx could be either
                # None or -1, for legacy reasons (which are now obviated).
                if idx is not None and idx >= 0:
                    updated_components.append(components[idx])
                else:
                    updated_components.append({'val': val, '_classname': 'Literal'})
            # InlineList was split out from InlineArray in v20. If is_json=True, then this is
            # actually an InlineList. If is_json=False, then we assume it's an InlineArray for now,
            # but it might actually be transformed into an InlineList when it is instantiated
            # (unfortunately, there is no way to disambiguate at this stage; see comments in
            # InlineArray._from_dict() for more details).
            updated_v: dict[str, Any] = {'_classname': 'InlineList' if v.get('is_json') else 'InlineArray'}
            if len(updated_components) > 0:
                updated_v['components'] = updated_components
            return k, updated_v
        if v['_classname'] == 'InlineDict':
            components = v.get('components')
            keys = []
            updated_components = []
            for key, idx, val in v['dict_items']:
                keys.append(key)
                if idx is not None and idx >= 0:
                    updated_components.append(components[idx])
                else:
                    updated_components.append({'val': val, '_classname': 'Literal'})
            updated_v = {'keys': keys, '_classname': 'InlineDict'}
            if len(updated_components) > 0:
                updated_v['components'] = updated_components
            return k, updated_v
    return None
