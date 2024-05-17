import logging
from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import Table

_logger = logging.getLogger('pixeltable')


def convert_13(engine: sql.engine.Engine) -> None:
    with engine.begin() as conn:
        for row in conn.execute(sql.select(Table)):
            id = row[0]
            md = row[2]
            updated_md = _update_md(md)
            if updated_md != md:
                _logger.info(f'Updating schema for table: {id}')
                conn.execute(sql.update(Table).where(Table.id == id).values(md=updated_md))


# Traverse the schema dictionary and replace instances of `ExplicitBatchedFunction` with
# `CallableFunction`. DB versions prior to 14 can't contain serialized batched functions,
# so this is all we need to do.
def _update_md(md: Any) -> Any:
    if isinstance(md, dict):
        updated_md = {}
        for k, v in md.items():
            if k == '_classpath' and v == 'pixeltable.func.batched_function.ExplicitBatchedFunction':
                updated_md[k] = 'pixeltable.func.callable_function.CallableFunction'
            else:
                updated_md[k] = _update_md(v)
        return updated_md
    elif isinstance(md, list):
        return [_update_md(v) for v in md]
    else:
        return md


register_converter(13, convert_13)
