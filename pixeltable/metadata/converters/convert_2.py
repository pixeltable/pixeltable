import dataclasses

import sqlalchemy as sql
import sqlalchemy.orm as orm

from pixeltable.metadata import schema, register_converter

def convert_2(engine: sql.engine.Engine) -> None:
    """
    Add view-related metadata.
    """
    return

register_converter(2, convert_2)
