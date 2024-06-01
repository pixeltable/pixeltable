import copy
import logging
from typing import Any, Callable, Optional

import sqlalchemy as sql

from pixeltable.metadata.schema import Table

__logger = logging.getLogger('pixeltable')


def convert_table_md(
    engine: sql.engine.Engine,
    column_md_updater: Optional[Callable[[dict], None]] = None,
    remote_md_updater: Optional[Callable[[dict], None]] = None,
    substitution_fn: Optional[Callable[[Any, Any], Optional[tuple[Any, Any]]]] = None
) -> None:
    with engine.begin() as conn:
        for row in conn.execute(sql.select(Table)):
            id = row[0]
            table_md = row[2]
            assert isinstance(table_md, dict)
            updated_table_md = copy.deepcopy(table_md)
            if column_md_updater is not None:
                __update_column_md(updated_table_md, column_md_updater)
            if remote_md_updater is not None:
                __update_remote_md(updated_table_md, remote_md_updater)
            if substitution_fn is not None:
                updated_table_md = __substitute_md_rec(updated_table_md, substitution_fn)
            if updated_table_md != table_md:
                __logger.info(f'Updating schema for table: {id}')
                conn.execute(sql.update(Table).where(Table.id == id).values(md=updated_table_md))


def __update_column_md(table_md: dict, column_md_updater: Callable[[dict], None]) -> None:
    columns_md = table_md['column_md']
    assert isinstance(columns_md, dict)
    for column_md in columns_md.values():
        column_md_updater(column_md)


def __update_remote_md(table_md: dict, remote_md_updater: Callable[[dict], None]) -> None:
    remotes_md = table_md['remotes']
    assert isinstance(remotes_md, list)
    for remote_md in remotes_md:
        remote_md_updater(remote_md)


def __substitute_md_rec(md: Any, substitution_fn: Callable[[Any, Any], Optional[tuple[Any, Any]]]) -> Any:
    if isinstance(md, dict):
        updated_md = {}
        for k, v in md.items():
            substitute = substitution_fn(k, v)
            if substitute is not None:
                updated_k, updated_v = substitute
                updated_md[updated_k] = updated_v
            else:
                updated_md[k] = __substitute_md_rec(v, substitution_fn)
        return updated_md
    elif isinstance(md, list):
        return [__substitute_md_rec(v, substitution_fn) for v in md]
    else:
        return md
