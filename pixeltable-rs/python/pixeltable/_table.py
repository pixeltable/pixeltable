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

    def _path(self) -> str:
        return _core.tbl_path(str(self._id))

    def _display_name(self) -> str:
        return _core.tbl_kind(str(self._id))

    def _display_str(self) -> str:
        return f'{self._display_name()} {self._path()!r}'

    def __repr__(self) -> str:
        kind = self._display_name()
        cols = ', '.join(f'{name}: {col_type!r}' for name, col_type in self._get_schema().items())
        return f'{kind[0].upper()}{kind[1:]} {self._path()!r} ({cols})'

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
        if len(kwargs) != 1:
            from pixeltable import exceptions as excs

            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'add_column() requires exactly one keyword argument of the form `col_name=col_type`; '
                f'got {len(kwargs)} arguments instead ({", ".join(kwargs.keys())})',
            )
        return self.add_columns(kwargs, if_exists=if_exists)

    @unimplemented('computed columns')
    def add_computed_column(self, **kwargs: Any) -> UpdateStatus: ...

    def drop_column(self, column: str | _core.Expr, if_not_exists: Literal['error', 'ignore'] = 'error') -> None:
        """Drop a column, given by name or by reference."""
        _core.tbl_drop_column(str(self._id), column, if_not_exists)

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column."""
        _core.tbl_rename_column(str(self._id), old_name, new_name)

    @unimplemented('alter_column()')
    def alter_column(self, column: Any, *, type_: Any) -> None: ...

    @unimplemented('alter_computed_column()')
    def alter_computed_column(self, *args: Any, **kwargs: Any) -> UpdateStatus: ...

    @unimplemented('embedding indexes')
    def add_embedding_index(self, *args: Any, **kwargs: Any) -> None: ...

    @unimplemented('B-tree indexes')
    def add_btree_index(self, *args: Any, **kwargs: Any) -> None: ...

    @unimplemented('embedding indexes')
    def drop_embedding_index(self, *args: Any, **kwargs: Any) -> None: ...

    @unimplemented('indexes')
    def drop_index(self, *args: Any, **kwargs: Any) -> None: ...

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
        """Insert rows: a list (or iterable) of dicts, or a single row given as keyword arguments."""
        if source_format is not None or schema_overrides is not None or return_rows:
            raise NotImplementedError(
                'insert() with source_format, schema_overrides or return_rows is not implemented yet in pixeltable-rs'
            )
        return _status_from_core(_core.tbl_insert(str(self._id), source, kwargs, on_error))

    @unimplemented('Table.compute()')
    def compute(self, source: Sequence[Any], /, **kwargs: Any) -> Any: ...

    @unimplemented('update()')
    def update(self, value_spec: dict[str, Any], where: Any = None, cascade: bool = True) -> UpdateStatus: ...

    @unimplemented('batch_update()')
    def batch_update(self, rows: Iterable[dict[str, Any]], **kwargs: Any) -> UpdateStatus: ...

    @unimplemented('recompute_columns()')
    def recompute_columns(self, *columns: Any, **kwargs: Any) -> UpdateStatus: ...

    @unimplemented('delete()')
    def delete(self, where: Any = None) -> UpdateStatus: ...

    @unimplemented('revert()')
    def revert(self) -> None: ...

    def get_versions(self, n: int | None = None) -> list[VersionMetadata]:
        """Metadata of the most recent versions of this table, newest first."""
        import datetime

        result: list[VersionMetadata] = []
        for version, created_at, change_type, ins, upd, dels, errs, schema_change in _core.tbl_versions(
            str(self._id), n
        ):
            result.append(
                VersionMetadata(
                    version=version,
                    created_at=datetime.datetime.fromtimestamp(created_at, tz=datetime.timezone.utc),
                    user=None,
                    change_type=change_type,
                    inserts=ins,
                    updates=upd,
                    deletes=dels,
                    errors=errs,
                    schema_change=schema_change,
                )
            )
        return result

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
    cls = InsertableTable if kind == 'table' else View
    return cls(UUID(tbl_id), version)


# attribute names of table handles cannot be used as column names
_core.register_reserved_names(sorted({name for name in dir(InsertableTable) + dir(View) if not name.startswith('_')}))
