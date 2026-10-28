"""User-facing types used for type annotations across the Pixeltable codebase."""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Union, get_args

from typing_extensions import TypeForm

if TYPE_CHECKING:
    from pixeltable import exprs
    from pixeltable.config import URI, ConfigVar


TableKind = Literal['table', 'view', 'snapshot']

FunctionKind = Literal['udf', 'aggregate', 'iterator', 'query']
_KIND_WIDTH = max(len(kind) for kind in get_args(FunctionKind))


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


class DirContents(TypedDict):
    """
    Represents the contents of a Pixeltable directory.
    """

    dirs: list[str]
    """List of directory paths contained in this directory."""
    tables: list[str]
    """List of table paths contained in this directory."""


class FunctionInfo(TypedDict):
    """A function in a [`FunctionList`][pixeltable.FunctionList]."""

    path: str
    """The import path, such as `pixeltable.functions.video.frame_iterator`."""
    name: str
    kind: FunctionKind
    call: str
    """How to call the function, such as `pxt.String.upper() -> pxt.String` for a method of string expressions, or
    `huggingface.clip(...)` for a function. The path of a function in `pixeltable.functions` is relative to that
    module."""
    signatures: list[str]
    """All signatures, with default values; a polymorphic function has more than one."""
    summary: str | None
    """The first paragraph of the docstring."""
    doc: str | None
    outputs: dict[str, str] | None
    """The output columns of an iterator; `None` for the other kinds, or if they depend on the arguments."""


class FunctionList(list[FunctionInfo]):
    """A list of functions, as returned by [`pxt.list_functions()`][pixeltable.list_functions].

    Printed, it shows one function per line; in a notebook, it renders as a table.
    """

    def __repr__(self) -> str:
        return '\n'.join(self._line(info) for info in self)

    def _line(self, info: FunctionInfo) -> str:
        line = f'{info["kind"]:<{_KIND_WIDTH}} {info["call"]}'
        if len(info['signatures']) > 1:
            line += f'  [{len(info["signatures"])} signatures]'
        if info['summary'] is not None:
            line += f'  # {info["summary"]}'
        return line

    def _repr_html_(self) -> str:
        rows = ''.join(
            f'<tr><td>{info["kind"]}</td><td><code>{html.escape(info["call"])}</code></td>'
            f'<td>{"" if info["summary"] is None else html.escape(info["summary"])}</td></tr>'
            for info in self
        )
        return f'<table><tr><th>kind</th><th>call</th><th>summary</th></tr>{rows}</table>'


class ColumnSpec(TypedDict, total=False):
    """
    Column specification, a dictionary representation of a column's schema.
    Exactly one of `type` or `value` must be included in the dictionary.
    """

    type: TypeForm
    """The column type (e.g., `pxt.Image`, `str`). Required unless `value` is specified."""
    value: 'exprs.Expr'
    """A Pixeltable expression for computed columns. Mutually exclusive with `type`."""
    primary_key: bool
    """Whether this column is part of the primary key. Defaults to `False`."""
    stored: bool
    """Whether to store the column data. Defaults to `True`."""
    media_validation: Literal['on_read', 'on_write']
    """When to validate media; `'on_read'` or `'on_write'`."""
    destination: str | Path | ConfigVar[URI]
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
