import inspect
import logging
from typing import Any

import cloudpickle  # type: ignore[import-untyped]
import sqlalchemy as sql

import pixeltable.func as func
import pixeltable.type_system as ts
from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import Function

_logger = logging.getLogger('pixeltable')


@register_converter(version=15)
def _(engine: sql.engine.Engine) -> None:
    with engine.begin() as conn:
        for row in conn.execute(sql.select(Function)):
            id, dir_id, md, binary_obj = row
            md['md'] = __update_md(md['md'], binary_obj)
            _logger.info(f'Updating function: {id}')
            conn.execute(sql.update(Function).where(Function.id == id).values(md=md))


def __update_md(orig_d: dict, binary_obj: bytes) -> Any:
    # construct dict produced by CallableFunction.to_store()
    py_fn = cloudpickle.loads(binary_obj)
    py_params = inspect.signature(py_fn).parameters
    return_type = ts.ColumnType.from_dict(orig_d['return_type'])
    params: list[func.Parameter] = []
    for name, col_type_dict, kind_int, is_batched in orig_d['parameters']:
        col_type = ts.ColumnType.from_dict(col_type_dict) if col_type_dict is not None else None
        default = py_params[name].default
        kind = inspect._ParameterKind(kind_int)  # is there a way to avoid referencing a private type?
        params.append(func.Parameter(name=name, col_type=col_type, kind=kind, default=default, is_batched=is_batched))
    is_batched = 'batch_size' in orig_d
    sig = func.Signature(return_type, params, is_batched=is_batched)
    d = {
        'signature': sig.as_dict(),
        'batch_size': orig_d['batch_size'] if is_batched else None,
    }
    return d
