from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping
from uuid import UUID, uuid4

from .table import Table

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import torch.utils.data

    from pixeltable import exprs, type_system as ts
    from pixeltable._query import Query, ResultCursor, ResultSet
    from pixeltable.func.function import Function
    from pixeltable.query_clauses import JoinType
    from pixeltable.types import ColumnSpec

    from ..exprs import ColumnRef
    from ..globals import TableDataSource
    from .path import Path as CatalogPath
    from .table_metadata import TableMetadata, VersionMetadata
    from .table_path import TableMdPath
    from .update_status import UpdateStatus


class TableProxy(Table):
    """Abstract base for handles to hosted tables, reached over RPC (delegated execution)."""

    _path_obj: CatalogPath

    def __init__(self, path: CatalogPath):
        super().__init__(uuid4())  # placeholder id; the real id is assigned by the dispatch server
        self._path_obj = path

    def _name(self) -> str:
        raise NotImplementedError

    def _dir_id(self) -> UUID | None:
        raise NotImplementedError

    def _path(self) -> 'CatalogPath':
        return self._path_obj

    @property
    def _tbl_path(self) -> 'TableMdPath':
        raise NotImplementedError

    def get_metadata(self) -> 'TableMetadata':
        raise NotImplementedError

    def __getattr__(self, name: str) -> 'exprs.ColumnRef':
        raise NotImplementedError

    def __getitem__(self, name: str) -> 'exprs.ColumnRef':
        raise NotImplementedError

    def list_views(self, *, recursive: bool = True) -> list[str]:
        raise NotImplementedError

    def select(self, *items: Any, **named_items: Any) -> 'Query':
        raise NotImplementedError

    def where(self, pred: 'exprs.Expr') -> 'Query':
        raise NotImplementedError

    def join(self, other: 'Table', *, on: 'exprs.Expr' | None = None, how: 'JoinType.LiteralType' = 'inner') -> 'Query':
        raise NotImplementedError

    def order_by(self, *items: 'exprs.Expr', asc: bool = True) -> 'Query':
        raise NotImplementedError

    def group_by(self, *items: 'exprs.Expr') -> 'Query':
        raise NotImplementedError

    def distinct(self) -> 'Query':
        raise NotImplementedError

    def limit(self, n: int, offset: int | None = None) -> 'Query':
        raise NotImplementedError

    def sample(
        self,
        n: int | None = None,
        n_per_stratum: int | None = None,
        fraction: float | None = None,
        seed: int | None = None,
        stratify_by: Any = None,
    ) -> Query:
        raise NotImplementedError

    def collect(self) -> 'ResultSet':
        raise NotImplementedError

    def cursor(self) -> 'ResultCursor':
        raise NotImplementedError

    def show(self, *args: Any, **kwargs: Any) -> 'ResultSet':
        raise NotImplementedError

    def head(self, *args: Any, **kwargs: Any) -> 'ResultSet':
        raise NotImplementedError

    def tail(self, *args: Any, **kwargs: Any) -> 'ResultSet':
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def columns(self) -> list[str]:
        raise NotImplementedError

    def _get_base_table(self) -> 'Table' | None:
        raise NotImplementedError

    def describe(self) -> None:
        raise NotImplementedError

    def to_pytorch_dataset(self, image_format: str = 'pt') -> 'torch.utils.data.IterableDataset':
        raise NotImplementedError

    def to_coco_dataset(self) -> Path:
        raise NotImplementedError

    def add_columns(
        self,
        schema: Mapping[str, type | ColumnSpec],
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> UpdateStatus:
        raise NotImplementedError

    def add_column(
        self,
        *,
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
        **kwargs: type | ColumnSpec,
    ) -> UpdateStatus:
        raise NotImplementedError

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
        **kwargs: exprs.Expr,
    ) -> UpdateStatus:
        raise NotImplementedError

    def drop_column(self, column: str | ColumnRef, if_not_exists: Literal['error', 'ignore'] = 'error') -> None:
        raise NotImplementedError

    def rename_column(self, old_name: str, new_name: str) -> None:
        raise NotImplementedError

    def add_embedding_index(
        self,
        column: str | ColumnRef,
        *,
        idx_name: str | None = None,
        embedding: Function | None = None,
        string_embed: Function | None = None,
        image_embed: Function | None = None,
        metric: Literal['cosine', 'ip', 'l2'] = 'cosine',
        precision: Literal['fp16', 'fp32'] = 'fp16',
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> None:
        raise NotImplementedError

    def drop_embedding_index(
        self,
        *,
        column: str | ColumnRef | None = None,
        idx_name: str | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        raise NotImplementedError

    def drop_index(
        self,
        *,
        column: str | ColumnRef | None = None,
        idx_name: str | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        raise NotImplementedError

    def insert(
        self,
        source: TableDataSource | None = None,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, ts.ColumnType] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        return_rows: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus:
        raise NotImplementedError

    def update(
        self,
        value_spec: dict[str, Any],
        where: 'exprs.Expr' | None = None,
        cascade: bool = True,
        return_rows: bool = False,
    ) -> UpdateStatus:
        raise NotImplementedError

    def batch_update(
        self,
        rows: Iterable[dict[str, Any]],
        cascade: bool = True,
        if_not_exists: Literal['error', 'ignore', 'insert'] = 'error',
        return_rows: bool = False,
    ) -> UpdateStatus:
        raise NotImplementedError

    def recompute_columns(
        self,
        *columns: str | ColumnRef,
        where: 'exprs.Expr' | None = None,
        errors_only: bool = False,
        cascade: bool = True,
    ) -> UpdateStatus:
        raise NotImplementedError

    def delete(self, where: 'exprs.Expr' | None = None) -> UpdateStatus:
        raise NotImplementedError

    def revert(self) -> None:
        raise NotImplementedError

    def external_stores(self) -> list[str]:
        raise NotImplementedError

    def unlink_external_stores(
        self, stores: str | list[str] | None = None, *, delete_external_data: bool = False, ignore_errors: bool = False
    ) -> None:
        raise NotImplementedError

    def sync(
        self, stores: str | list[str] | None = None, *, export_data: bool = True, import_data: bool = True
    ) -> UpdateStatus:
        raise NotImplementedError

    def get_versions(self, n: int | None = None) -> list[VersionMetadata]:
        raise NotImplementedError

    def history(self, n: int | None = None) -> pd.DataFrame:
        raise NotImplementedError
