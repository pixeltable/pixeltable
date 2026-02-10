from pathlib import Path
from typing import Literal, TypedDict

from pixeltable import exprs

"""Types used for type annotations across the Pixeltable codebase."""


class ColumnSpec(TypedDict, total=False):
    """
    Column specification, a dictionary representation of a column's schema.
    Exactly one of `type` or `value` must be included in the dictionary.
    """

    type: type
    """The column type (e.g., `pxt.Image`, `str`). Required unless `value` is specified."""
    value: exprs.Expr
    """A Pixeltable expression for computed columns. Mutually exclusive with `type`."""
    primary_key: bool
    """Whether this column is part of the primary key. Defaults to `False`."""
    stored: bool
    """Whether to store the column data. Defaults vary by column type."""
    media_validation: Literal['on_read', 'on_write']
    """When to validate media; `'on_read'` or `'on_write'`."""
    destination: str | Path
    """
    Destination for storing computed output files. Only applicable for computed columns.
    Can be:
        - A local pathname (such as `path/to/outputs/`), or
        - The URI of an object store (such as `s3://my-bucket/outputs/`).
    """
