import logging
from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import Table

_logger = logging.getLogger('pixeltable')


@register_converter(version=13)
def _(engine: sql.engine.Engine) -> None:
    with engine.begin() as conn:
        for row in conn.execute(sql.select(Table.id, Table.md)):
            id = row[0]
            md = row[1]
            updated_md = __update_md(md)
            if updated_md != md:
                _logger.info(f'Updating schema for table: {id}')
                conn.execute(sql.update(Table).where(Table.id == id).values(md=updated_md))


# Traverse the schema dictionary and replace instances of `ExplicitBatchedFunction` with
# `CallableFunction`. DB versions prior to 14 can't contain serialized batched functions,
# so this is all we need to do.
def __update_md(md: Any) -> Any:
    if isinstance(md, dict):
        updated_md = {}
        for k, v in md.items():
            if k == '_classpath' and v == 'pixeltable.func.batched_function.ExplicitBatchedFunction':
                updated_md[k] = 'pixeltable.func.callable_function.CallableFunction'
            else:
                updated_md[k] = __update_md(v)
        return updated_md
    elif isinstance(md, list):
        return [__update_md(v) for v in md]
    else:
        return md
