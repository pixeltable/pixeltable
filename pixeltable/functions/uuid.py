"""
Pixeltable UDFs for `UUIDType`.
"""

import uuid

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.utils.code import local_public_names


@pxt.udf
def uuid4() -> uuid.UUID:
    """
    Generate a random UUID (version 4).
    """
    return uuid.uuid4()


@uuid4.to_sql
def _() -> sql.ColumnElement:
    return sql.func.gen_random_uuid()  # Generates uuid version 4


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
