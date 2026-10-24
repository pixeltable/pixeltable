"""`pxt.udf(table)`: a function that computes a table's computed columns for given data column values (upstream
`udf.from_table()`)."""

from __future__ import annotations

import inspect
from typing import Any

from pixeltable import _core

from ._expr_template import ExprTemplate, ExprTemplateFunction
from ._signature import Parameter, Signature


def from_table(tbl: Any, return_value: Any, description: str | None) -> ExprTemplateFunction:
    """The data columns of the table (and its bases) are the parameters (optional if nullable); the result is a dict
    of all columns, or `return_value`, computed from them."""
    expr, param_specs = _core.table_udf(str(tbl._id), return_value)
    params = [
        Parameter(
            name,
            col_type,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            _core.expr_from_object(None) if col_type.nullable else None,
        )
        for name, col_type in param_specs
    ]
    md = tbl.get_metadata()
    if description is None:
        description = md['comment']
        if description is None or len(description) == 0:
            description = f"UDF for table '{md['name']}'"
    argstring = '\n'.join(f'    {param.name}: of type `{param.col_type}`' for param in params)
    template = ExprTemplate(expr, Signature(expr.col_type, params))
    fn = ExprTemplateFunction([template], name=md['name'])
    fn.__doc__ = f'{description}\n\nArgs:\n{argstring}'
    return fn
