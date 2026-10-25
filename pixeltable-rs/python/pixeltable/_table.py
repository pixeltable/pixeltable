"""Table handles: `pxt.Table`, `pxt.InsertableTable`, `pxt.View`, and directory handles.

A handle holds only the table id (and a pinned version for `get_table('t:3')`); every operation is forwarded to the
backend by id, so handles stay valid across `reload_catalog()` and see renames and moves.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence
from uuid import UUID

import pandas as pd

from pixeltable import _core
from pixeltable._metadata import TableMetadata, VersionMetadata
from pixeltable._unimplemented import unimplemented
from pixeltable._update_status import UpdateStatus, _status_from_core

if TYPE_CHECKING:
    from pixeltable._query import Query, ResultSet
    from pixeltable._row import RowBatch


class SchemaObject:
    """Base class of directories and tables."""

    _id: UUID

    def __init__(self, obj_id: UUID):
        self._id = obj_id


class Dir(SchemaObject):
    """A directory handle (returned by `pxt.create_dir()`)."""

    def _display_name(self) -> str:
        return 'directory'

    def __repr__(self) -> str:
        return f'Dir({self._id})'


class _TablePath:
    """What upstream's `TablePath` answers about the leaf of a handle's table path (a pure snapshot's leaf is the
    table it pins)."""

    def __init__(self, tbl_name: str, is_snapshot: bool, is_view: bool, is_component_view: bool):
        self._tbl_name = tbl_name
        self._is_snapshot = is_snapshot
        self._is_view = is_view
        self._is_component_view = is_component_view

    def tbl_name(self) -> str:
        return self._tbl_name

    def is_snapshot(self) -> bool:
        return self._is_snapshot

    def is_view(self) -> bool:
        return self._is_view

    def is_component_view(self) -> bool:
        return self._is_component_view


class Table(SchemaObject):
    """
    A handle to a table, view, or snapshot: the primary interface for table operations (queries, insertions,
    schema changes).
    """

    _version: int | None

    def __init__(self, tbl_id: UUID, version: int | None = None):
        super().__init__(tbl_id)
        self._version = version

    # ---- identity

    @property
    def _tbl_ref(self) -> tuple[str, int | None]:
        return (str(self._id), self._version)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Table) and self._id == other._id and self._version == other._version

    def __hash__(self) -> int:
        return hash((self._id, self._version))

    def _name(self) -> str:
        return _core.tbl_name(str(self._id))

    @property
    def _tbl_path(self) -> _TablePath:
        """The table path backing this handle (upstream `Table._tbl_path`)."""
        return _TablePath(*_core.tbl_path_leaf(str(self._id), self._version))

    def _path(self) -> str:
        return _core.tbl_path(str(self._id))

    def _display_name(self) -> str:
        return _core.tbl_kind(str(self._id))

    def _display_str(self) -> str:
        return f'{self._display_name()} {self._path()!r}'

    def __repr__(self) -> str:
        from pixeltable import _describe

        return _describe.to_string(_core.tbl_describe(str(self._id), self._version))

    def _repr_html_(self) -> str:
        from pixeltable import _describe

        return _describe.to_html(_core.tbl_describe(str(self._id), self._version))

    def _get_version(self) -> int:
        """The version this handle reads: its pinned version, else the table's current version."""
        return self._version if self._version is not None else _core.tbl_version(str(self._id))

    def _get_schema(self) -> dict[str, Any]:
        return _core.tbl_schema(str(self._id), self._version)

    # ---- column references

    def __getattr__(self, name: str) -> _core.Expr:
        if name.startswith('_'):
            raise AttributeError(f'{type(self).__name__!r} object has no attribute {name!r}')
        return _core.tbl_column_ref(str(self._id), self._version, name)

    def __getitem__(self, name: str) -> _core.Expr:
        return _core.tbl_column_ref(str(self._id), self._version, name)

    def columns(self) -> list[str]:
        """The names of the columns of this table."""
        return _core.tbl_columns(str(self._id), self._version)

    def get_metadata(self) -> TableMetadata:
        """Metadata of this table (see [TableMetadata][pixeltable.TableMetadata])."""
        return _core.tbl_metadata(str(self._id), self._version)  # type: ignore[return-value]

    def list_views(self, *, recursive: bool = True) -> list[str]:
        """Paths of the views and snapshots of this table."""
        return _core.tbl_list_views(str(self._id), recursive)

    def get_base_table(self) -> Table | None:
        md = self.get_metadata()
        if md['base'] is None:
            return None
        from pixeltable._globals import get_table

        return get_table(md['base'])

    def describe(self) -> None:
        print(repr(self))

    # ---- queries

    def _query(self) -> Query:
        from pixeltable._query import Query

        return Query._from_table(self)

    def select(self, *items: Any, **named_items: Any) -> Query:
        """Select columns or expressions from this table. See [`Query.select`][pixeltable.Query.select]."""
        return self._query().select(*items, **named_items)

    def where(self, pred: _core.Expr) -> Query:
        """Filter rows from this table. See [`Query.where`][pixeltable.Query.where]."""
        return self._query().where(pred)

    def join(self, other: Table, *, on: Any = None, how: str = 'inner') -> Query:
        return self._query().join(other, on=on, how=how)  # type: ignore[arg-type]

    def order_by(self, *items: _core.Expr, asc: bool = True) -> Query:
        return self._query().order_by(*items, asc=asc)

    def group_by(self, *items: Any) -> Query:
        return self._query().group_by(*items)

    def distinct(self) -> Query:
        return self._query().distinct()

    def limit(self, n: int, offset: int | None = None) -> Query:
        return self._query().limit(n, offset)

    def sample(self, *args: Any, **kwargs: Any) -> Query:
        return self._query().sample(*args, **kwargs)

    def collect(self) -> ResultSet:
        return self._query().collect()

    def cursor(self) -> Any:
        return self._query().cursor()

    def show(self, n: int = 20) -> ResultSet:
        return self._query().show(n)

    def head(self, n: int = 10) -> ResultSet:
        return self._query().head(n)

    def tail(self, n: int = 10) -> ResultSet:
        return self._query().tail(n)

    def count(self) -> int:
        """The number of rows in this table."""
        return self._query().count()

    def to_pytorch_dataset(self, image_format: str = 'pt') -> Any:
        return self._query().to_pytorch_dataset(image_format)

    def to_coco_dataset(self) -> Path:
        return self._query().to_coco_dataset()

    # ---- schema changes

    def add_columns(
        self,
        schema: Mapping[str, Any],
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> UpdateStatus:
        """Add columns to the table."""
        _core.tbl_add_columns(str(self._id), dict(schema), if_exists)
        return UpdateStatus()

    def add_column(
        self,
        *,
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
        **kwargs: Any,
    ) -> UpdateStatus:
        """Add a column, given as `name=type` or `name=ColumnSpec`."""
        _core.tbl_add_column(str(self._id), kwargs, if_exists)
        return UpdateStatus()

    def add_computed_column(
        self,
        *,
        stored: bool | None = None,
        destination: str | Path | None = None,
        custom_metadata: Any = None,
        comment: str = '',
        print_stats: bool = False,
        on_error: Literal['abort', 'ignore'] = 'abort',
        if_exists: Literal['error', 'ignore', 'replace'] = 'error',
        **kwargs: Any,
    ) -> UpdateStatus:
        """Add a computed column, given as `name=expression`."""
        return _status_from_core(
            _core.tbl_add_computed_column(
                str(self._id), kwargs, stored, destination, custom_metadata, comment, on_error, if_exists
            )
        )

    def drop_column(self, column: str | _core.Expr, if_not_exists: Literal['error', 'ignore'] = 'error') -> None:
        """Drop a column, given by name or by reference."""
        _core.tbl_drop_column(str(self._id), column, if_not_exists)

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column."""
        _core.tbl_rename_column(str(self._id), old_name, new_name)

    def alter_column(self, column: str | _core.Expr, *, type_: Any) -> None:
        """Change the type of a column (currently: make a column nullable)."""
        _core.tbl_alter_column(str(self._id), self._version, column, type_)

    def alter_computed_column(self, *, recompute: bool = True, cascade: bool = True, **kwargs: Any) -> UpdateStatus:
        """Replace the value expression of a computed column, given as `name=expression`."""
        return _status_from_core(
            _core.tbl_alter_computed_column(str(self._id), self._version, kwargs, recompute, cascade)
        )

    def add_embedding_index(
        self,
        column: str | _core.Expr,
        *,
        idx_name: str | None = None,
        embedding: Any = None,
        string_embed: Any = None,
        image_embed: Any = None,
        audio_embed: Any = None,
        video_embed: Any = None,
        document_embed: Any = None,
        metric: Literal['cosine', 'ip', 'l2'] = 'cosine',
        precision: Literal['fp16', 'fp32'] = 'fp16',
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> None:
        """Add an embedding index to a column (see upstream `Table.add_embedding_index()`)."""
        _core.tbl_add_embedding_index(
            str(self._id),
            column,
            idx_name,
            embedding,
            string_embed,
            image_embed,
            audio_embed,
            video_embed,
            document_embed,
            metric,
            precision,
            if_exists,
        )

    def add_btree_index(
        self, column: str | _core.Expr, *, idx_name: str | None = None, if_exists: Literal['error', 'ignore'] = 'error'
    ) -> None:
        """Add a B-tree index to a column."""
        _core.tbl_add_btree_index(str(self._id), column, idx_name, if_exists)

    def drop_embedding_index(
        self,
        *,
        column: str | _core.Expr | None = None,
        idx_name: str | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        """Drop an embedding index, given by column (with exactly one embedding index) or by name."""
        _core.tbl_drop_index(str(self._id), column, idx_name, True, if_not_exists)

    def drop_index(
        self,
        *,
        column: str | _core.Expr | None = None,
        idx_name: str | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        """Drop an index, given by column (with exactly one index) or by name."""
        _core.tbl_drop_index(str(self._id), column, idx_name, False, if_not_exists)

    # ---- data changes

    def insert(
        self,
        source: Any = None,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, Any] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        return_rows: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus:
        """Insert rows: a list (or iterable) of dicts, a list of pydantic models, a table or query, or a single row
        given as keyword arguments."""
        return _status_from_core(
            _core.tbl_insert(
                str(self._id),
                self._version,
                source,
                kwargs,
                on_error,
                return_rows,
                source_format,
                schema_overrides,
            )
        )

    def compute(
        self,
        source: Sequence[Any],
        /,
        *,
        outputs: Sequence[Any] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
    ) -> RowBatch:
        """Materialize the computed columns of this table for the given input rows, without storing them."""
        from pixeltable._row import RowBatch

        data, col_types, errors = _core.tbl_compute(str(self._id), self._version, source, outputs, on_error)
        return RowBatch(data, col_types, errors=errors)

    def update(
        self,
        value_spec: dict[str, Any],
        where: _core.Expr | None = None,
        cascade: bool = True,
        return_rows: bool = False,
    ) -> UpdateStatus:
        """Update the rows that satisfy `where` (all rows without it), setting columns to literals or expressions."""
        return _status_from_core(
            _core.tbl_update(str(self._id), self._version, value_spec, where, cascade, return_rows)
        )

    def batch_update(
        self,
        rows: Iterable[dict[str, Any]],
        cascade: bool = True,
        if_not_exists: Literal['error', 'ignore', 'insert'] = 'error',
        return_rows: bool = False,
    ) -> UpdateStatus:
        """Update rows identified by their primary key (or `_rowid`), each with its own values."""
        return _status_from_core(
            _core.tbl_batch_update(str(self._id), self._version, rows, cascade, if_not_exists, return_rows)
        )

    def recompute_columns(
        self,
        *columns: str | _core.Expr,
        where: _core.Expr | None = None,
        errors_only: bool = False,
        cascade: bool = True,
    ) -> UpdateStatus:
        """Recompute the values of computed columns (and, with `cascade`, of the columns that depend on them)."""
        return _status_from_core(
            _core.tbl_recompute_columns(str(self._id), self._version, list(columns), where, errors_only, cascade)
        )

    def delete(self, where: _core.Expr | None = None) -> UpdateStatus:
        """Delete the rows that satisfy `where` (all rows without it)."""
        return _status_from_core(_core.tbl_delete(str(self._id), self._version, where))

    def revert(self) -> None:
        """Revert the table to its previous version."""
        _core.tbl_revert(str(self._id), self._version)

    def get_versions(self, n: int | None = None) -> list[VersionMetadata]:
        """Metadata of the most recent versions of this table, newest first."""
        import datetime

        return [
            VersionMetadata(
                version=version,
                created_at=datetime.datetime.fromtimestamp(created_at, tz=datetime.timezone.utc),
                user=user,
                change_type=change_type,
                inserts=ins,
                updates=upd,
                deletes=dels,
                errors=errs,
                schema_change=schema_change,
            )
            for version, created_at, user, change_type, ins, upd, dels, errs, schema_change in _core.tbl_versions(
                str(self._id), n
            )
        ]

    def history(self, n: int | None = None) -> pd.DataFrame:
        """A human-readable version history of this table."""
        return pd.DataFrame(self.get_versions(n))


class InsertableTable(Table):
    """A base table."""


class View(Table):
    """A view or snapshot."""

    @classmethod
    def validate_view_query(cls, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError('validate_view_query() is not implemented yet in pixeltable-rs')

    @classmethod
    def select_list_to_additional_columns(cls, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('select_list_to_additional_columns() is not implemented yet in pixeltable-rs')


def _make_table(tbl_id: str, kind: str, version: int | None = None) -> Table:
    # a handle pinned to a version is a (anonymous) snapshot, as upstream
    cls = InsertableTable if kind == 'table' and version is None else View
    return cls(UUID(tbl_id), version)


# attribute names of table handles cannot be used as column names
_core.register_reserved_names(sorted({name for name in dir(InsertableTable) + dir(View) if not name.startswith('_')}))
