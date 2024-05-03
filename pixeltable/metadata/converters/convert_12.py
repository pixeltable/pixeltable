import sqlalchemy as sql

from pixeltable.metadata import register_converter


def convert_12(engine: sql.engine.Engine) -> None:
    # Version 12 schemas will load in version 13 without modification
    # (but not vice versa).
    pass


register_converter(12, convert_12)
