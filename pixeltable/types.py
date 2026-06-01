"""User-facing types used for type annotations across the Pixeltable codebase."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Union

if TYPE_CHECKING:
    from pixeltable import exprs


TableKind = Literal['table', 'view', 'snapshot', 'replica']


class DirectoryNode(TypedDict):
    """A directory entry in a [`TreeNode`][pixeltable.TreeNode] tree."""

    name: str
    path: str
    kind: Literal['directory']
    entries: list['TreeNode']


class TableNode(TypedDict):
    """A table/view/snapshot/replica entry in a [`TreeNode`][pixeltable.TreeNode] tree."""

    name: str
    path: str
    kind: TableKind
    version: int | None
    error_count: int
    """Cumulative error count as recorded in table's history."""
    base: str | None
    """Path of the immediate base table for views/snapshots; None for plain tables."""


TreeNode = Union[DirectoryNode, TableNode]


class ColumnSpec(TypedDict, total=False):
    """
    Column specification, a dictionary representation of a column's schema.
    Exactly one of `type` or `value` must be included in the dictionary.
    """

    type: type
    """The column type (e.g., `pxt.Image`, `str`). Required unless `value` is specified."""
    value: 'exprs.Expr'
    """A Pixeltable expression for computed columns. Mutually exclusive with `type`."""
    primary_key: bool
    """Whether this column is part of the primary key. Defaults to `False`."""
    stored: bool
    """Whether to store the column data. Defaults to `True`."""
    media_validation: Literal['on_read', 'on_write']
    """When to validate media; `'on_read'` or `'on_write'`."""
    destination: str | Path
    """
    Destination for storing computed output files. Only applicable for computed columns.
    Can be:

        - A local pathname (such as `path/to/outputs/`), or
        - The URI of an object store (such as `s3://my-bucket/outputs/`).
    """
    custom_metadata: Any
    """User-defined metadata to associate with the column."""
    comment: str
    """Optional comment for the column. Displayed in .describe() output."""
