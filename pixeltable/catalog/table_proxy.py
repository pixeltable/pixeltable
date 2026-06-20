from __future__ import annotations

import builtins
import dataclasses
import threading
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping
from uuid import UUID

import pixeltable.exceptions as excs

from ..exprs import ColumnRef
from .globals import normalize_schema
from .path import Path as CatalogPath
from .table import Table
from .table_path import TableMdPath, TablePathKey

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pydantic
    import torch.utils.data

    from pixeltable import exprs, type_system as ts
    from pixeltable.func.function import Function
    from pixeltable.service.proxy_client import ProxyClient
    from pixeltable.types import ColumnSpec

    from ..globals import TableDataSource
    from .globals import TableVersionMd
    from .table_metadata import TableMetadata, VersionMetadata
    from .update_status import UpdateStatus


class TableProxy(Table):
    """A handle to a hosted table, reached over RPC (delegated execution).

    Backed by a TableMdPath obtained from the dispatch server, so schema introspection works locally.

    Thread-safe.
    """

    _key: TablePathKey  # immutable version identity; shared read-only across threads
    _local: threading.local  # per-thread tbl_md_path; seeded by the constructing thread, else lazily fetched
    _client: 'ProxyClient'
    _catalog_uri: CatalogPath  # the hosted catalog this proxy lives in; stamped onto md paths for query routing

    def __init__(self, tbl_md_path: TableMdPath, client: 'ProxyClient'):
        super().__init__(tbl_md_path.tbl_id)
        self._key = tbl_md_path.key()
        self._local = threading.local()
        self._local.tbl_md_path = tbl_md_path  # seed the constructing thread
        self._client = client
        self._catalog_uri = tbl_md_path.catalog_uri

    @property
    def _tbl_md_path(self) -> TableMdPath:
        md_path = getattr(self._local, 'tbl_md_path', None)
        if md_path is None:
            md_path = self._fetch_md_path()  # a thread other than the one that constructed the proxy
            self._local.tbl_md_path = md_path
        return md_path

    def _fetch_md_path(self) -> TableMdPath:
        # reach the table by id; the leaf's effective version pins the right version for a snapshot proxy
        leaf = self._key.leaf
        md = self._client.send_request(
            'CatalogBase', 'get_table_by_id', {'tbl_id': leaf.tbl_id, 'version': leaf.effective_version}
        )
        if md is None:
            raise excs.NotFoundError(excs.ErrorCode.TABLE_NOT_FOUND, f'Table not found: {leaf.tbl_id}')
        return TableMdPath.from_md(md, self._catalog_uri)

    def _refresh_md_path(self, md: list[TableVersionMd]) -> None:
        self._local.tbl_md_path = TableMdPath.from_md(md, self._catalog_uri)

    def _dispatch(self, method: str, args: dict[str, Any]) -> Any:
        return self._client.dispatch_table_method(
            method,
            args,
            path_key=self._key,
            get_snapshot_key=lambda: self._tbl_md_path.snapshot_key(),
            refresh=self._refresh_md_path,
        )

    def _name(self) -> str:
        return self._tbl_md_path.tbl_name()

    def __eq__(self, other: object) -> bool:
        # two proxy handles denote the same table when they share a table id; built fresh per call (unlike
        # local handles, which the catalog caches), so identity comparison would be wrong
        return isinstance(other, TableProxy) and other._id == self._id

    def __hash__(self) -> int:
        return hash(self._id)

    def _get_version(self) -> int | None:
        return self._tbl_md_path.version()

    def _is_versioned(self) -> bool:
        return self._tbl_md_path.is_versioned()

    def _dir_id(self) -> UUID | None:
        raise NotImplementedError

    def _path(self) -> CatalogPath:
        # the server returns the in-db path (org/db dropped on the wire); rebase onto this proxy's catalog
        server_path = self._dispatch('_path', {})
        return dataclasses.replace(server_path, org=self._catalog_uri.org, db=self._catalog_uri.db)

    @property
    def _query_path(self) -> TableMdPath:
        # the path that supplies queryable rows/columns; same as the identity path except for a pure
        # snapshot, which has no physical table and is queried via its base (see ViewProxy)
        return self._tbl_md_path

    @property
    def _tbl_path(self) -> TableMdPath:
        return self._query_path

    def get_metadata(self) -> 'TableMetadata':
        return self._dispatch('get_metadata', {})

    def __getattr__(self, name: str) -> 'exprs.ColumnRef':
        if name.startswith('_'):
            # an internal/dunder attribute miss is never a column; raise now to avoid recursing via _tbl_md_path
            raise AttributeError(name)
        col_md = self._query_path.get_column_md_by_name(name)
        if col_md is None:
            raise AttributeError(f'Unknown column: {name}')
        return ColumnRef(col_md)

    def __getitem__(self, name: str) -> 'exprs.ColumnRef':
        return getattr(self, name)

    def list_views(self, *, recursive: bool = True) -> list[str]:
        return self._dispatch('list_views', {'recursive': recursive})

    def columns(self) -> list[str]:
        return [cvmd.name for cvmd in self._query_path.column_md() if cvmd.name is not None]

    def _get_base_table(self) -> 'Table' | None:
        from .insertable_table_proxy import InsertableTableProxy
        from .view_proxy import ViewProxy

        base = self._tbl_md_path.base
        if base is None:
            return None
        return (ViewProxy if base.is_view() else InsertableTableProxy)(base, self._client)

    def describe(self) -> None:
        if getattr(builtins, '__IPYTHON__', False):
            from IPython.display import Markdown, display

            display(Markdown(self._repr_html_()))
        else:
            print(repr(self))

    def __repr__(self) -> str:
        # send the uri as a string: proxy_protocol drops org/db from wire Paths, which is exactly what describe needs
        return self._dispatch('describe', {'catalog_uri': self._catalog_uri.uri})['str']

    def _repr_html_(self) -> str:
        return self._dispatch('describe', {'catalog_uri': self._catalog_uri.uri})['html']

    def to_pytorch_dataset(self, image_format: str = 'pt') -> 'torch.utils.data.IterableDataset': ...

    def to_coco_dataset(self) -> Path: ...

    def add_columns(
        self,
        schema: Mapping[str, type | ColumnSpec],
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> UpdateStatus:
        self._validate_column_schema(schema)
        return self._dispatch('add_columns', {'schema': normalize_schema(schema), 'if_exists': if_exists})

    def add_column(
        self,
        *,
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
        **kwargs: type | ColumnSpec,
    ) -> UpdateStatus:
        if len(kwargs) != 1:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'add_column() requires exactly one keyword argument of the form `col_name=col_type`; '
                f'got {len(kwargs)} arguments instead ({", ".join(kwargs.keys())})',
            )
        self._validate_column_schema(kwargs)
        return self._dispatch('add_column', {'columns': normalize_schema(kwargs), 'if_exists': if_exists})

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
        return self._dispatch(
            'add_computed_column',
            {
                'columns': kwargs,
                'stored': stored,
                'destination': destination,
                'custom_metadata': custom_metadata,
                'comment': comment,
                'print_stats': print_stats,
                'on_error': on_error,
                'if_exists': if_exists,
            },
        )

    def drop_column(self, column: str | ColumnRef, if_not_exists: Literal['error', 'ignore'] = 'error') -> None:
        self._dispatch('drop_column', {'column': column, 'if_not_exists': if_not_exists})

    def rename_column(self, old_name: str, new_name: str) -> None:
        self._dispatch('rename_column', {'old_name': old_name, 'new_name': new_name})

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
        self._dispatch(
            'add_embedding_index',
            {
                'column': column,
                'idx_name': idx_name,
                'embedding': embedding,
                'string_embed': string_embed,
                'image_embed': image_embed,
                'metric': metric,
                'precision': precision,
                'if_exists': if_exists,
            },
        )

    def drop_embedding_index(
        self,
        *,
        column: str | ColumnRef | None = None,
        idx_name: str | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        self._dispatch('drop_embedding_index', {'column': column, 'idx_name': idx_name, 'if_not_exists': if_not_exists})

    def drop_index(
        self,
        *,
        column: str | ColumnRef | None = None,
        idx_name: str | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        self._dispatch('drop_index', {'column': column, 'idx_name': idx_name, 'if_not_exists': if_not_exists})

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
    ) -> UpdateStatus: ...

    def compute(
        self,
        source: Sequence[dict[str, Any]] | Sequence[pydantic.BaseModel],
        /,
        *,
        on_error: Literal['abort', 'ignore'] = 'abort',
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def update(
        self,
        value_spec: dict[str, Any],
        where: 'exprs.Expr' | None = None,
        cascade: bool = True,
        return_rows: bool = False,
    ) -> UpdateStatus:
        self._validate_update_value_spec(value_spec)
        self._validate_where(where)
        return self._dispatch(
            'update', {'value_spec': value_spec, 'where': where, 'cascade': cascade, 'return_rows': return_rows}
        )

    def batch_update(
        self,
        rows: Iterable[dict[str, Any]],
        cascade: bool = True,
        if_not_exists: Literal['error', 'ignore', 'insert'] = 'error',
        return_rows: bool = False,
    ) -> UpdateStatus:
        return self._dispatch(
            'batch_update',
            {'rows': list(rows), 'cascade': cascade, 'if_not_exists': if_not_exists, 'return_rows': return_rows},
        )

    def recompute_columns(
        self,
        *columns: str | ColumnRef,
        where: 'exprs.Expr' | None = None,
        errors_only: bool = False,
        cascade: bool = True,
    ) -> UpdateStatus:
        return self._dispatch(
            'recompute_columns',
            {'columns': list(columns), 'where': where, 'errors_only': errors_only, 'cascade': cascade},
        )

    def revert(self) -> None:
        self._dispatch('revert', {})

    def external_stores(self) -> list[str]: ...

    def unlink_external_stores(
        self, stores: str | list[str] | None = None, *, delete_external_data: bool = False, ignore_errors: bool = False
    ) -> None: ...

    def sync(
        self, stores: str | list[str] | None = None, *, export_data: bool = True, import_data: bool = True
    ) -> UpdateStatus: ...

    def get_versions(self, n: int | None = None) -> list[VersionMetadata]:
        return self._dispatch('get_versions', {'n': n})
