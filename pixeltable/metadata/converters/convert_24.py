from typing import Any, Optional

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=24)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=__substitute_md)


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    from pixeltable import func
    from pixeltable.func.globals import resolve_symbol

    if (
        isinstance(v, dict)
        and '_classpath' in v
        and v['_classpath']
        in (
            'pixeltable.func.callable_function.CallableFunction',
            'pixeltable.func.aggregate_function.AggregateFunction',
            'pixeltable.func.expr_template_function.ExprTemplateFunction',
        )
    ):
        if 'path' in v:
            assert 'signature' not in v
            f = resolve_symbol(__substitute_path(v['path']))
            assert isinstance(f, func.Function)
            v['signature'] = f.signatures[0].as_dict()
        return k, v

    if isinstance(v, dict) and '_classname' in v and v['_classname'] == 'FunctionCall':
        # Correct an older serialization mechanism where Expr elements of FunctionCall args and
        # kwargs were indicated with idx == -1 rather than None. This was fixed for InlineList
        # and InlineDict back in convert_20, but not for FunctionCall.
        assert 'args' in v and isinstance(v['args'], list)
        assert 'kwargs' in v and isinstance(v['kwargs'], dict)
        v['args'] = [(None, arg) if idx == -1 else (idx, arg) for idx, arg in v['args']]
        v['kwargs'] = {k: (None, arg) if idx == -1 else (idx, arg) for k, (idx, arg) in v['kwargs'].items()}
        return k, v

    return None


def __substitute_path(path: str) -> str:
    # Starting with version 25, function signatures are preserved in metadata. To migrate from older
    # versions, it's necessary to resolve the function symbol to get the signature. The following
    # adjustment is necessary for function names that are stored in db artifacts of version < 25, but
    # have changed in some version > 25.
    if path in ('pixeltable.functions.huggingface.clip_text', 'pixeltable.functions.huggingface.clip_image'):
        return 'pixeltable.functions.huggingface.clip'
    return path
