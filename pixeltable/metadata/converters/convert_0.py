import dataclasses

import sqlalchemy as sql
import sqlalchemy.orm as orm

from pixeltable.metadata import schema, register_converter

def convert_0(engine: sql.engine.Engine) -> None:
    """Sample converter for non-existent version 0

    This converter backfills a new field 'f: int' in DirMd.
    """
    with orm.Session(engine, future=True) as session:
        for dir in session.query(schema.Dir).all():
            dir_md = schema.DirMd(name=dir.md['name'], f=0)
            dir.md = dataclasses.asdict(dir_md)
        session.commit()

register_converter(0, convert_0)
