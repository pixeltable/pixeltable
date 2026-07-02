"""User-facing types used for type annotations across the Pixeltable codebase."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Union

if TYPE_CHECKING:
    from pixeltable import exprs, func


TableKind = Literal['table', 'view', 'snapshot']


class DirectoryNode(TypedDict):
    """A directory entry in a [`TreeNode`][pixeltable.TreeNode] tree."""

    name: str
    path: str
    kind: Literal['directory']
    entries: list['TreeNode']


class TableNode(TypedDict):
    """A table/view/snapshot entry in a [`TreeNode`][pixeltable.TreeNode] tree."""

    name: str
    path: str
    kind: TableKind
    version: int | None
    error_count: int
    """Cumulative error count as recorded in table's history."""
    base: str | None
    """Path of the immediate base table for views/snapshots; None for plain tables."""


TreeNode = Union[DirectoryNode, TableNode]


class EmbeddingIndexSpec(TypedDict, total=False):
    """A serializable embedding-index specification, used to carry a model's declared embedding indexes to the
    catalog that creates the table (so they can be created within the same table-creation unit). `column` is the
    name of the column to index; the embedding functions and parameters mirror `Table.add_embedding_index`."""

    idx_name: str | None
    column: str
    metric: Literal['cosine', 'ip', 'l2']
    precision: Literal['fp16', 'fp32']
    embed: 'func.Function | None'
    string_embed: 'func.Function | None'
    image_embed: 'func.Function | None'
    audio_embed: 'func.Function | None'
    video_embed: 'func.Function | None'
    document_embed: 'func.Function | None'


class DirContents(TypedDict):
    """
    Represents the contents of a Pixeltable directory.
    """

    dirs: list[str]
    """List of directory paths contained in this directory."""
    tables: list[str]
    """List of table paths contained in this directory."""


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
