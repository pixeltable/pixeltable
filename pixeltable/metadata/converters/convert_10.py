import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import Table

def convert_10(engine: sql.engine.Engine) -> None:
    # TODO As a first cut, this is implemented using raw SQL statements.
    # It would perhaps be better to implement it using idiomatic SQLAlchemy.
    default_table_attrs = '{"attrs": {"comment": "", "num_retained_versions": 10}}'
    with engine.connect() as conn:
        # Because `parameters` wasn't actually used for anything,
        # we can simply delete it without any data loss.
        conn.execute(sql.text("UPDATE tables SET md = md - 'parameters';"))
        # Add `attrs` to all elements of schematableversions.
        conn.execute(sql.text(f"UPDATE tableschemaversions SET md = md || '{default_table_attrs}';"))
        conn.commit()
    return


register_converter(10, convert_10)
