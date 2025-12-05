"""
Pixeltable UDFs for `UUIDType`.
"""

import uuid

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.utils.code import local_public_names


@pxt.udf(is_method=False)
def make_uuid() -> uuid.UUID:
    """
    Generate a random UUID (version 4).

    Equivalent to [`uuid.uuid4()`](https://docs.python.org/3/library/uuid.html#uuid.uuid4).
    """
    return uuid.uuid4()


@make_uuid.to_sql
def _() -> sql.ColumnElement:
    return sql.func.gen_random_uuid()


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
