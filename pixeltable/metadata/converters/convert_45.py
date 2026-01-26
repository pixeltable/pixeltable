from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md

    # old ColumnMd (inside TableMd.column_md as a dict of col id to ColumnMd)
    # id: int
    # schema_version_add: int
    # schema_version_drop: int | None
    # col_type: dict

    # # if True, is part of the primary key
    # is_pk: bool

    # # if set, this is a computed column
    # value_expr: dict | None

    # # if True, the column is present in the stored table
    # stored: bool | None

    # # If present, the URI for the destination for column values
    # destination: str | None = None

    
    # new ColumnMd
    # id: int
    # schema_version_add: int
    # schema_version_drop: int | None

    # # if True, the column is present in the stored table
    # stored: bool | None

    
    # old SchemaColumn (inside SchemaVersionMd.columns as a dict of col id to SchemaColumn)
    # pos: int
    # name: str

    # # media validation strategy of this particular media column; if not set, TableMd.media_validation applies
    # # stores column.MediaValiation.name.lower()
    # media_validation: str | None


    # new SchemaColumn:
    # # pos is set for user-visible columns, None otherwise
    # pos: int | None
    # name: str

    # col_type: dict
    # # if True, is part of the primary key
    # is_pk: bool
    # # if set, this is a computed column
    # value_expr: dict | None

    # # media validation strategy of this particular media column; if not set, TableMd.media_validation applies
    # # stores column.MediaValiation.name.lower()
    # media_validation: str | None

    # # If present, the URI for the destination for column values
    # destination: str | None = None


@register_converter(version=45)
def _(engine: sql.engine.Engine) -> None:
    """
    TODO
    """
    # convert_table_md(engine, substitution_fn=_substitution_fn)
    pass


def _substitution_fn(key: str | None, value: Any) -> tuple[str | None, Any] | None:
    pass