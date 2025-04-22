import datetime
from typing import Any, Optional

import sqlalchemy as sql

import pixeltable.type_system as ts
from pixeltable.metadata import register_converter, schema
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=19)
def _(engine: sql.engine.Engine) -> None:
    # Convert all timestamp literals to aware datetimes
    convert_table_md(engine, substitution_fn=__update_timestamp_literals)

    # Convert all timestamp columns to TIMESTAMPTZ. (This conversion will take place in the database
    # default time zone, which is what we want, since in versions <= 19 they were naive timestamps.)
    with engine.begin() as conn:
        tables = conn.execute(sql.select(schema.Table.id, schema.Table.md))
        for id, md in tables:
            store_prefix = 'view' if md['view_md'] is not None else 'tbl'
            store_name = f'{store_prefix}_{id.hex}'
            column_md = md['column_md']
            timestamp_cols = [
                col_id for col_id, col in column_md.items() if col['col_type']['_classname'] == 'TimestampType'
            ]
            for col_id in timestamp_cols:
                conn.execute(sql.text(f'ALTER TABLE {store_name} ALTER COLUMN col_{col_id} TYPE TIMESTAMPTZ'))


def __update_timestamp_literals(k: Any, v: Any) -> Optional[tuple[Any, Any]]:
    if isinstance(v, dict) and 'val_t' in v:
        # It's a literal with an explicit 'val_t' field. In version 19 this can only mean a
        # timestamp literal, which (in version 19) is stored in the DB as a naive datetime.
        # We convert it to an aware datetime, stored in UTC.
        assert v['_classname'] == 'Literal'
        assert v['val_t'] == ts.ColumnType.Type.TIMESTAMP.name
        assert isinstance(v['val'], str)
        dt = datetime.datetime.fromisoformat(v['val'])
        assert dt.tzinfo is None  # In version 19 all timestamps are naive
        dt_utc = dt.astimezone(datetime.timezone.utc)
        v['val'] = dt_utc.isoformat()
        return k, v
    return None
