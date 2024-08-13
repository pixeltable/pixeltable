import datetime
from typing import TYPE_CHECKING, Any, Optional

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md

if TYPE_CHECKING:
    from pixeltable.catalog import Column, TableVersion


# @register_converter(version=18)
def _(engine: sql.engine.Engine) -> None:
    from pixeltable.catalog import Catalog

    with engine.begin() as conn:
        for tbl_version in Catalog.get().tbl_versions.values():
            for col in tbl_version.cols:
                if col.col_type.is_timestamp_type():
                    __update_timestamp_col(conn, tbl_version, col)
    convert_table_md(engine, substitution_fn=__update_timestamp_literals)


def __update_timestamp_col(conn: sql.Connection, tbl_version: 'TableVersion', col: 'Column') -> None:
    from pixeltable.catalog import Column, TableVersion

    sa_tbl = tbl_version.store_tbl.sa_tbl
    sa_col = col.sa_col
    conn.execute(
        sql.text(f'ALTER TABLE {sa_tbl.name} ALTER COLUMN {sa_col.name} TYPE TIMESTAMPTZ;')
    )


def __update_timestamp_literals(k: Any, v: Any) -> Optional[tuple[Any, Any]]:
    if isinstance(v, dict) and 'val_t' in v:
        # It's a literal with an explicit 'val_t' field. In version 18 this can only mean a
        # timestamp literal.
        assert v['val_t'] == pxt.ColumnType.Type.TIMESTAMP.name
        assert isinstance(v['val'], str)
        dt = datetime.datetime.fromisoformat(v['val'])
        assert dt.tzinfo is None  # In version 18 all timestamps are naive
        dt_utc = dt.astimezone(datetime.timezone.utc)
        v['val'] = dt_utc.isoformat()
        return k, v
