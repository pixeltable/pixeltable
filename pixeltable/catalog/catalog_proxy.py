from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping
from uuid import UUID

from pixeltable.env import Env

from .catalog_base import CatalogBase
from .insertable_table_proxy import InsertableTableProxy
from .table_path import TableMdPath
from .view_proxy import ViewProxy

if TYPE_CHECKING:
    from pixeltable import exprs, func
    from pixeltable._query import Query
    from pixeltable.plan import SampleClause
    from pixeltable.service.proxy_client import ProxyClient
    from pixeltable.types import ColumnSpec, EmbeddingIndexSpec

    from .dir import Dir
    from .globals import DirEntry, IfExistsParam, IfNotExistsParam, MediaValidation, TableVersionMd
    from .path import Path
    from .table import Table
    from .table_path import TablePath


class CatalogProxy(CatalogBase):
    """
    An implementation of CatalogBase that delegates to a hosted catalog.
    """

    _catalog_uri: Path
    _client: ProxyClient

    def __init__(self, catalog_uri: Path, client: ProxyClient):
        self._catalog_uri = catalog_uri
        self._client = client

    def _make_table(self, md: list[TableVersionMd], is_anon_snapshot: bool) -> Table:
        tbl_id = UUID(md[0].tbl_md.tbl_id)
        Env.get().record_tbl_catalog_uri(tbl_id, self._catalog_uri)

        tbl_md_path = TableMdPath.from_md(md, is_anon_snapshot=is_anon_snapshot, catalog_uri=self._catalog_uri)

        if md[0].tbl_md.view_md is not None or is_anon_snapshot:
            return ViewProxy(tbl_id, is_anon_snapshot, tbl_md_path, self._client)
        else:
            return InsertableTableProxy(tbl_id, tbl_md_path, self._client)

    def run_query(self, method: str, query_dict: dict, **extra: Any) -> Any:
        """Execute a Query method against the hosted catalog."""
        return self._client.send_request('Query', method, {'query': query_dict, **extra})

    def create_table(
        self,
        path: Path,
        schema: dict[str, ColumnSpec],
        if_exists: IfExistsParam,
        primary_key: list[str] | None,
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        create_default_idxs: bool,
        is_versioned: bool,
    ) -> tuple[Table, bool]:
        args = {
            'path': path,
            'schema': schema,
            'if_exists': if_exists,
            'primary_key': primary_key,
            'comment': comment,
            'custom_metadata': custom_metadata,
            'media_validation': media_validation,
            'create_default_idxs': create_default_idxs,
            'is_versioned': is_versioned,
        }
        md, was_created = self._client.send_request('CatalogBase', 'create_table', args)
        # effective_version=None: this is a live table
        return self._make_table(md, is_anon_snapshot=False), was_created

    def create_view(
        self,
        path: Path,
        base: TablePath,
        select_list: list[tuple[exprs.Expr, str | None]] | None,
        where: exprs.Expr | None,
        sample_clause: SampleClause | None,
        additional_columns: Mapping[str, ColumnSpec] | None,
        is_snapshot: bool,
        create_default_idxs: bool,
        iterator: func.GeneratingFunctionCall | None,
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        if_exists: IfExistsParam,
    ) -> tuple[Table, bool]:
        args = {
            'path': path,
            'base': base,
            'select_list': select_list,
            'where': where,
            'sample_clause': sample_clause,
            'additional_columns': additional_columns or {},
            'is_snapshot': is_snapshot,
            'create_default_idxs': create_default_idxs,
            'iterator': iterator,
            'comment': comment,
            'custom_metadata': custom_metadata,
            'media_validation': media_validation,
            'if_exists': if_exists,
        }
        md, was_created = self._client.send_request('CatalogBase', 'create_view', args)
        return self._make_table(md, is_anon_snapshot=False), was_created

    def create_from_model(
        self,
        path: Path,
        columns: dict[str, ColumnSpec],
        display_name: str,
        create_default_idxs: bool,
        media_validation: MediaValidation,
        comment: str | None,
        custom_metadata: Any,
        iterator: func.GeneratingFunctionCall | None,
        base: 'Query | None',
        embedding_idxs: list['EmbeddingIndexSpec'],
    ) -> tuple[Table, bool]:
        args = {
            'path': path,
            'columns': columns,
            'display_name': display_name,
            'create_default_idxs': create_default_idxs,
            'media_validation': media_validation,
            'comment': comment,
            'custom_metadata': custom_metadata,
            'iterator': iterator,
            'base': base.as_dict() if base is not None else None,
            'embedding_idxs': embedding_idxs,
        }
        md, was_created = self._client.send_request('CatalogBase', 'create_from_model', args)
        return self._make_table(md, is_anon_snapshot=False), was_created

    def get_table(self, path: Path, if_not_exists: IfNotExistsParam) -> Table | None:
        md = self._client.send_request('CatalogBase', 'get_table', {'path': path, 'if_not_exists': if_not_exists})
        if md is None:
            return None
        if path.version is not None:
            # we requested an anonymous snapshot; md is leaf-first, so the snapshot's version is md[0]'s
            assert path.version == md[0].version_md.version
        return self._make_table(md, is_anon_snapshot=path.version is not None)

    def get_table_by_id(
        self, tbl_id: UUID, version: int | None = None, ignore_if_dropped: bool = False
    ) -> Table | None:
        md = self._client.send_request(
            'CatalogBase',
            'get_table_by_id',
            {'tbl_id': tbl_id, 'version': version, 'ignore_if_dropped': ignore_if_dropped},
        )
        return None if md is None else self._make_table(md, is_anon_snapshot=version is not None)

    def drop_table(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None:
        self._client.send_request(
            'CatalogBase', 'drop_table', {'path': path, 'if_not_exists': if_not_exists, 'force': force}
        )

    def move(self, path: Path, new_path: Path, if_exists: IfExistsParam, if_not_exists: IfNotExistsParam) -> None:
        self._client.send_request(
            'CatalogBase',
            'move',
            {'path': path, 'new_path': new_path, 'if_exists': if_exists, 'if_not_exists': if_not_exists},
        )

    def get_dir_contents(
        self, dir_path: Path, recursive: bool = False, with_error_counts: bool = False
    ) -> dict[str, DirEntry]:
        return self._client.send_request(
            'CatalogBase',
            'get_dir_contents',
            {'dir_path': dir_path, 'recursive': recursive, 'with_error_counts': with_error_counts},
        )

    def create_dir(self, path: Path, if_exists: IfExistsParam, parents: bool) -> Dir:
        return self._client.send_request(
            'CatalogBase', 'create_dir', {'path': path, 'if_exists': if_exists, 'parents': parents}
        )

    def drop_dir(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None:
        self._client.send_request(
            'CatalogBase', 'drop_dir', {'path': path, 'if_not_exists': if_not_exists, 'force': force}
        )
