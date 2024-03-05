import sqlalchemy as sql

from pixeltable.metadata.schema import Table, TableSchemaVersion
from pixeltable.metadata import register_converter


def convert_10(engine: sql.engine.Engine) -> None:
    default_table_attrs = {"attrs": {"comment": "", "num_retained_versions": 10}}
    with engine.begin() as conn:
        # Because `parameters` wasn't actually used for anything,
        # we can simply delete it without any data loss.
        conn.execute(sql.update(Table).values(md=Table.md - 'parameters'))
        # Add `attrs` to all elements of schematableversions.
        conn.execute(sql.update(TableSchemaVersion).values(md=Table.md.concat(default_table_attrs)))
    return


register_converter(10, convert_10)
