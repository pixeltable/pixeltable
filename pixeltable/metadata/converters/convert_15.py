import inspect
import logging
from typing import Any

import cloudpickle  # type: ignore[import-untyped]
import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import legacy_functions

_logger = logging.getLogger(__name__)


@register_converter(version=15)
def _(conn: sql.Connection) -> None:
    # select explicit columns (not SELECT *) so a future column addition/reorder can't shift the unpacking
    for row in conn.execute(sql.select(legacy_functions.c.id, legacy_functions.c.md, legacy_functions.c.binary_obj)):
        id, md, binary_obj = row
        md['md'] = __update_md(md['md'], binary_obj)
        _logger.info(f'Updating function: {id}')
        conn.execute(sql.update(legacy_functions).where(legacy_functions.c.id == id).values(md=md))


def __update_md(orig_d: dict, binary_obj: bytes) -> Any:
    # The pickled body is needed only to recover parameter default values. A function pickled under an older Python
    # can fail to unpickle on a newer one (cloudpickle code-object incompatibility); such a pickle-backed function
    # can no longer be used or persisted anyway, so migrate its signature without defaults instead of failing.
    try:
        py_params: Any = inspect.signature(cloudpickle.loads(binary_obj), eval_str=True).parameters
    except Exception:
        py_params = None
    return_type = orig_d['return_type']
    params: list[dict] = []
    for name, col_type_dict, kind_int, is_batched in orig_d['parameters']:
        default = py_params[name].default if py_params is not None else inspect.Parameter.empty
        kind = inspect._ParameterKind(kind_int)
        params.append(
            {
                'name': name,
                'col_type': col_type_dict,
                'kind': str(kind),
                'is_batched': is_batched,
                'has_default': default is not inspect.Parameter.empty,
                'default': None if default is inspect.Parameter.empty else default,
            }
        )
    is_batched = 'batch_size' in orig_d
    d = {
        'signature': {'return_type': return_type, 'parameters': params, 'is_batched': is_batched},
        'batch_size': orig_d['batch_size'] if is_batched else None,
    }
    return d
