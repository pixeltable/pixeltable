"""
Pixeltable UDFs for `UUID`.
"""

import uuid

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.utils.code import local_public_names
from pixeltable.utils.uuid import uuid7 as _uuid7_impl


@pxt.udf(is_deterministic=False)
def uuid4() -> uuid.UUID:
    """
    Generate a random UUID (version 4).

    Equivalent to [`uuid.uuid4()`](https://docs.python.org/3/library/uuid.html#uuid.uuid4).
    """
    return uuid.uuid4()


@uuid4.to_sql
def _() -> sql.ColumnElement:
    return sql.func.gen_random_uuid()  # Generates uuid version 4


@pxt.udf(is_deterministic=False)
def uuid7() -> uuid.UUID:
    """
    Generate a time-based UUID.

    Equivalent to [`uuid.uuid7()`](https://docs.python.org/3/library/uuid.html#uuid.uuid7).
    """
    return _uuid7_impl()


@pxt.udf(is_method=True)
def to_string(u: uuid.UUID) -> str:
    """
    Convert a UUID to its string representation.

    Args:
        u: The UUID to convert.

    Returns:
        The string representation of the UUID, in the form `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.

    Example:
        Convert the UUID column `id` in an existing table `tbl` to a string:

        >>> tbl.add_computed_column(id_string=to_string(tbl.id))
    """
    return str(u)


@to_string.to_sql
def _(u: sql.ColumnElement) -> sql.ColumnElement:
    return u.cast(sql.Text)


@pxt.udf(is_method=True)
def hex(u: uuid.UUID) -> str:
    """
    Convert a UUID to its hexadecimal representation.

    Equivalent to [`uuid.hex`](https://docs.python.org/3/library/uuid.html#uuid.UUID.hex).

    Args:
        u: The UUID to convert.

    Returns:
        The hexadecimal representation of the UUID, as a 32-character string of hex digits.

    Example:
        Convert the UUID column `id` in an existing table `tbl` to a hexadecimal string:

        >>> tbl.add_computed_column(id_hex=hex(tbl.id))
    """
    return u.hex


@hex.to_sql
def _(u: sql.ColumnElement) -> sql.ColumnElement:
    # Convert UUID to text and remove hyphens
    return sql.func.replace(u.cast(sql.Text), '-', '')


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
