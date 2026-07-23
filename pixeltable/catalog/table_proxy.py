from __future__ import annotations

import builtins
import dataclasses
import threading
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping
from uuid import UUID

import pixeltable.exceptions as excs
from pixeltable import type_system as ts

from ..exprs import ColumnRef
from .globals import normalize_schema
from .path import Path as CatalogPath
from .table import Table
from .table_path import TableMdPath, TablePathKey, TableVersionKey

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pydantic
    import torch.utils.data

    from pixeltable import exprs
    from pixeltable.func.function import Function
    from pixeltable.service.proxy_client import ProxyClient
    from pixeltable.types import ColumnSpec

    from ..globals import TableDataSource
    from .globals import TableVersionMd
    from .table_metadata import TableMetadata, VersionMetadata
    from .update_status import UpdateStatus


class TableProxy(Table):
    """A handle to a hosted table.

    Backed by a TableMdPath obtained from the dispatch server, so schema introspection works locally.

    Thread-safe.
    """

    # self._id + self._effective_version uniquely identify the schema entity
    _effective_version: int | None

    _path_key: TablePathKey  # reflects tbl_md_path; immutable; shared read-only across threads
    _local: threading.local  # per-thread tbl_md_path; seeded by the constructing thread, else lazily fetched
    _client: 'ProxyClient'
    _catalog_uri: CatalogPath  # the hosted catalog this proxy lives in; stamped onto md paths for query routing

    def __init__(self, id: UUID, effective_version: int | None, tbl_md_path: TableMdPath, client: 'ProxyClient'):
        super().__init__(id)
        self._effective_version = effective_version
        self._path_key = tbl_md_path.key()
        self._local = threading.local()
        self._local.tbl_md_path = tbl_md_path
        self._client = client
        self._catalog_uri = tbl_md_path.catalog_uri

    @property
    def _tbl_md_path(self) -> TableMdPath:
        md_path = getattr(self._local, 'tbl_md_path', None)
        if md_path is None:
            md_path = self._fetch_md_path()
            self._local.tbl_md_path = md_path
        return md_path

    @property
    def _is_anon_snapshot(self) -> bool:
        return self._effective_version is not None

    def _fetch_md_path(self) -> TableMdPath:
        # reach the table by id; the leaf's effective version pins the right version for a snapshot proxy
        md = self._client.send_request(
            'CatalogBase', 'get_table_by_id', {'tbl_id': self._id, 'version': self._effective_version}
        )
        if md is None:
            raise excs.table_was_dropped(self._id)
        return TableMdPath.from_md(md, self._is_anon_snapshot, self._catalog_uri)

    def _refresh_md_path(self, md: list[TableVersionMd]) -> None:
        self._local.tbl_md_path = TableMdPath.from_md(md, self._is_anon_snapshot, catalog_uri=self._catalog_uri)

    def _snapshot_key(self) -> TablePathKey:
        # Bound to the proxy (stable), so it re-reads _tbl_md_path on every call. refresh() swaps in a new
        # tbl_md_path during a mutation CAS retry; a key bound to the pre-refresh path would never converge.
        return self._tbl_md_path.snapshot_key()

    @staticmethod
    def _dispatch_args(local_vars: dict[str, Any]) -> dict[str, Any]:
        """Build a server-dispatch args dict from a method's locals().

        Must be called as the first statement of the method (before any other local is bound), so that local_vars
        contains exactly the method's parameters. Callers then mutate the returned dict for any args that need
        transforming before dispatch.
        """
        return {k: v for k, v in local_vars.items() if k != 'self'}

    def _dispatch(self, method: str, args: dict[str, Any]) -> Any:
        tbl_key = TableVersionKey(self._id, self._effective_version)
        return self._client.dispatch_table_method(
            method,
            args,
            path_key=TablePathKey((tbl_key,)),
            get_snapshot_key=self._snapshot_key,
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
    def _tbl_path(self) -> TableMdPath:
        return self._tbl_md_path

    def _rebase_path(self, path_str: str) -> str:
        # the server emits in-db paths (org/db dropped on the wire); stamp this proxy's catalog back on
        local_path = CatalogPath.parse(path_str, allow_versioned_path=True)
        return str(dataclasses.replace(local_path, org=self._catalog_uri.org, db=self._catalog_uri.db))

    def get_metadata(self) -> 'TableMetadata':
        output = self._dispatch('get_metadata', {})
        # every path-valued field comes back as an in-db path; rebase each onto this proxy's catalog
        output['path'] = self._rebase_path(output['path'])
        if output['base'] is not None:
            output['base'] = self._rebase_path(output['base'])
        return output

    def __getattr__(self, name: str) -> 'exprs.ColumnRef':
        if name.startswith('_'):
            # an internal/dunder attribute miss is never a column; raise now to avoid recursing via _tbl_md_path
            raise AttributeError(name)
        col_md = self._tbl_path.get_column_md_by_name(name)
        if col_md is None:
            raise AttributeError(f'Unknown column: {name}')
        return ColumnRef(col_md, self._tbl_path.is_validate_on_read(col_md))

    def __getitem__(self, name: str) -> 'exprs.ColumnRef':
        return getattr(self, name)

    def list_views(self, *, recursive: bool = True) -> list[str]:
        return self._dispatch('list_views', self._dispatch_args(locals()))

    def columns(self) -> list[str]:
        return [col_md.name for col_md in self._tbl_path.column_md() if col_md.name is not None]

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
        create_default_idxs: bool = False,
    ) -> UpdateStatus:
        bound_args = self._dispatch_args(locals())
        self._check_mutable('add columns to')
        self._validate_column_schema(schema)
        bound_args['schema'] = normalize_schema(schema)
        return self._dispatch('add_columns', bound_args)

    def add_column(
        self,
        *,
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
        create_default_idx: bool = False,
        **kwargs: type | ColumnSpec,
    ) -> UpdateStatus:
        self._check_single_column_kwarg('add_column', '`col_name=col_type`', kwargs)
        bound_args = self._dispatch_args(locals())
        self._check_mutable('add columns to')
        self._validate_column_schema(kwargs)
        bound_args['columns'] = normalize_schema(bound_args.pop('kwargs'))
        return self._dispatch('add_column', bound_args)

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
        create_default_idx: bool = False,
        **kwargs: exprs.Expr,
    ) -> UpdateStatus:
        bound_args = self._dispatch_args(locals())
        self._check_single_column_kwarg('add_computed_column', '`col_name=col_type` or `col_name=expression`', kwargs)
        self._check_mutable('add columns to')
        bound_args['columns'] = bound_args.pop('kwargs')
        return self._dispatch('add_computed_column', bound_args)

    def drop_column(self, column: str | ColumnRef, if_not_exists: Literal['error', 'ignore'] = 'error') -> None:
        bound_args = self._dispatch_args(locals())
        self._check_mutable('drop columns from')
        self._dispatch('drop_column', bound_args)

    def rename_column(self, old_name: str, new_name: str) -> None:
        bound_args = self._dispatch_args(locals())
        self._check_mutable('rename columns of')
        self._dispatch('rename_column', bound_args)

    def alter_column(self, column: str | ColumnRef, *, type_: type) -> None:
        bound_args = self._dispatch_args(locals())
        self._check_mutable('alter columns of')

        # normalize type_ to a ColumnType so that it can be serialized
        bound_args['type_'] = ts.ColumnType.normalize_type(type_, nullable_default=True, allow_builtin_types=False)
        self._dispatch('alter_column', bound_args)

    def add_btree_index(
        self, column: str | ColumnRef, *, idx_name: str | None = None, if_exists: Literal['error', 'ignore'] = 'error'
    ) -> None:
        bound_args = self._dispatch_args(locals())
        self._check_mutable('add an index to')
        self._dispatch('add_btree_index', bound_args)

    def add_embedding_index(
        self,
        column: str | ColumnRef,
        *,
        idx_name: str | None = None,
        embedding: Function | None = None,
        string_embed: Function | None = None,
        image_embed: Function | None = None,
        audio_embed: Function | None = None,
        video_embed: Function | None = None,
        document_embed: Function | None = None,
        metric: Literal['cosine', 'ip', 'l2'] = 'cosine',
        precision: Literal['fp16', 'fp32'] = 'fp16',
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> None:
        bound_args = self._dispatch_args(locals())
        self._check_mutable('add an index to')
        self._validate_embedding_args(embedding, string_embed, image_embed)
        self._dispatch('add_embedding_index', bound_args)

    def drop_embedding_index(
        self,
        *,
        column: str | ColumnRef | None = None,
        idx_name: str | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        bound_args = self._dispatch_args(locals())
        self._check_mutable('drop an index from')
        self._dispatch('drop_embedding_index', bound_args)

    def drop_index(
        self,
        *,
        column: str | ColumnRef | None = None,
        idx_name: str | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        bound_args = self._dispatch_args(locals())
        self._check_mutable('drop an index from')
        self._dispatch('drop_index', bound_args)

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
        bound_args = self._dispatch_args(locals())
        self._check_mutable('update')
        self._validate_update_value_spec(value_spec)
        self._validate_where(where)
        return self._dispatch('update', bound_args)

    def batch_update(
        self,
        rows: Iterable[dict[str, Any]],
        cascade: bool = True,
        if_not_exists: Literal['error', 'ignore', 'insert'] = 'error',
        return_rows: bool = False,
    ) -> UpdateStatus:
        bound_args = self._dispatch_args(locals())
        self._check_mutable('update')
        bound_args['rows'] = list(rows)
        return self._dispatch('batch_update', bound_args)

    def recompute_columns(
        self,
        *columns: str | ColumnRef,
        where: 'exprs.Expr' | None = None,
        errors_only: bool = False,
        cascade: bool = True,
    ) -> UpdateStatus:
        bound_args = self._dispatch_args(locals())
        self._check_mutable('recompute columns of')
        bound_args['columns'] = list(columns)
        return self._dispatch('recompute_columns', bound_args)

    def revert(self) -> None:
        self._check_mutable('revert')
        self._dispatch('revert', {})

    def get_versions(self, n: int | None = None) -> list[VersionMetadata]:
        return self._dispatch('get_versions', self._dispatch_args(locals()))
