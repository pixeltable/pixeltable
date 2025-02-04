from typing import Any, Optional

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=18)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=__substitute_md)


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    # Migrate a few changed function names
    if k == 'path' and v == 'pixeltable.functions.string.str_format':
        return 'path', 'pixeltable.functions.string.format'
    if k == 'path' and v.startswith('pixeltable.functions.pil.image'):
        return 'path', v.replace('pixeltable.functions.pil.image', 'pixeltable.functions.image')
    # Migrate deprecated `ImageMemberAccess` expressions to `FunctionCall`s
    if isinstance(v, dict) and '_classname' in v and v['_classname'] == 'ImageMemberAccess':
        member_name = v['member_name']
        new_v = {
            'fn': {
                'path': f'pixeltable.functions.image.{member_name}',
                '_classpath': 'pixeltable.func.callable_function.CallableFunction',
            },
            'args': [[0, None]],
            'kwargs': {},
            '_classname': 'FunctionCall',
            'components': v['components'],
            'group_by_stop_idx': 0,
            'group_by_start_idx': 0,
            'order_by_start_idx': 1,
        }
        return k, new_v
    return None
