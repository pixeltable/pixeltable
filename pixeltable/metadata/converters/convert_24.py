from typing import Any, Optional
import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=24)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=__substitute_md)


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    if (isinstance(v, dict) and
        '_classpath' in v and
        v['_classpath'] == 'pixeltable.func.expr_template_function.ExprTemplateFunction'):
        # Modify the metadata for ExprTemplateFunction to reflect the new multi-template pattern
        if 'path' not in v:
            assert 'expr' in v and 'signature' in v and 'name' in v
            v = {
                'name': v['name'],
                'templates': [
                    {
                        'expr': v['expr'],
                        'signature': v['signature'],
                    }
                ],
                '_classpath': v['_classpath'],
            }
            return k, v

    if isinstance(v, dict) and '_classname' in v and v['_classname'] == 'FunctionCall':
        # Correct an older serialization mechanism where Expr elements of FunctionCall args and
        # kwargs were indicated with idx == -1 rather than None. This was fixed for InlineList
        # and InlineDict back in convert_20, but not for FunctionCall.
        assert 'args' in v and isinstance(v['args'], list)
        assert 'kwargs' in v and isinstance(v['kwargs'], dict)
        v['args'] = [
            (None, arg) if idx == -1 else (idx, arg)
            for idx, arg in v['args']
        ]
        v['kwargs'] = {
            k: (None, arg) if idx == -1 else (idx, arg)
            for k, (idx, arg) in v['kwargs'].items()
        }
        return k, v

    return None
