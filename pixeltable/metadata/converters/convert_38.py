from typing import Any, Optional

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=38)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=__substitute_md)


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    if k == 'col_mapping':
        assert isinstance(v, list)
        return k, [__col_mapping_entry(e) for e in v]
    if k == 'stored_proxies':
        assert isinstance(v, list)
        return k, [__stored_proxies_entry(e) for e in v]
    return None


def __col_mapping_entry(e: list) -> list:
    assert isinstance(e, list)
    assert isinstance(e[0], dict)
    assert isinstance(e[1], str)
    return [__col_handle(e[0]), e[1]]


def __stored_proxies_entry(e: list) -> list:
    assert isinstance(e, list)
    assert isinstance(e[0], dict)
    assert isinstance(e[1], dict)
    return [__col_handle(e[0]), __col_handle(e[1])]


def __col_handle(e: dict) -> dict:
    return {'tbl_version': {'id': e['tbl_id'], 'effective_version': None}, 'col_id': e['col_id']}
