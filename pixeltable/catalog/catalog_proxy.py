from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from pixeltable import exceptions as excs
from pixeltable.env import Env

from .catalog_base import CatalogBase
from .globals import normalize_schema
from .insertable_table_proxy import InsertableTableProxy
from .table_path import TableMdPath
from .view_proxy import ViewProxy

if TYPE_CHECKING:
    from uuid import UUID

    from pixeltable import exprs, func
    from pixeltable.plan import SampleClause
    from pixeltable.service.proxy_client import ProxyClient
    from pixeltable.types import ColumnSpec

    from .dir import Dir
    from .globals import DirEntry, IfExistsParam, IfNotExistsParam, MediaValidation, TableVersionMd
    from .path import Path
    from .table import Table
    from .table_path import TablePath


class CatalogProxy(CatalogBase):
    """
    An implementation of CatalogBase that delegates to a hosted catalog .
    """

    def __init__(self, catalog_uri: str, client: ProxyClient):
        self._catalog_uri = catalog_uri
        self._client = client

    def _make_table(self, md: list[TableVersionMd]) -> Table:
        tbl_md_path = TableMdPath.from_md(md, self._catalog_uri)
        # record which catalog this table belongs to, so a ColumnRef into it can resolve against this proxy
        Env.get().record_tbl_catalog_uri(tbl_md_path.tbl_id, self._catalog_uri)
        if tbl_md_path.is_view():
            return ViewProxy(tbl_md_path, self._client)
        return InsertableTableProxy(tbl_md_path, self._client)

    def run_query(self, method: str, query_dict: dict, **extra: Any) -> Any:
        """Execute a Query method against the hosted catalog."""
        return self._client.send_request('Query', method, {'query': query_dict, **extra})

    def create_table(
        self,
        path: Path,
        schema: dict[str, type | ColumnSpec | exprs.Expr],
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
            'schema': normalize_schema(schema),
            'if_exists': if_exists,
            'primary_key': primary_key,
            'comment': comment,
            'custom_metadata': custom_metadata,
            'media_validation': media_validation,
            'create_default_idxs': create_default_idxs,
            'is_versioned': is_versioned,
        }
        md, was_created = self._client.send_request('CatalogBase', 'create_table', args)
        return self._make_table(md), was_created

    def create_view(
        self,
        path: Path,
        base: TablePath,
        select_list: list[tuple[exprs.Expr, str | None]] | None,
        where: exprs.Expr | None,
        sample_clause: SampleClause | None,
        additional_columns: Mapping[str, type | ColumnSpec | exprs.Expr] | None,
        is_snapshot: bool,
        create_default_idxs: bool,
        iterator: func.GeneratingFunctionCall | None,
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        if_exists: IfExistsParam,
    ) -> Table:
        if iterator is not None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, 'Hosted iterator views are not supported yet.'
            )
        args = {
            'path': path,
            'base': base,
            'select_list': select_list,
            'where': where,
            'sample_clause': sample_clause,
            'additional_columns': normalize_schema(additional_columns) if additional_columns is not None else {},
            'is_snapshot': is_snapshot,
            'create_default_idxs': create_default_idxs,
            'iterator': iterator,
            'comment': comment,
            'custom_metadata': custom_metadata,
            'media_validation': media_validation,
            'if_exists': if_exists,
        }
        md = self._client.send_request('CatalogBase', 'create_view', args)
        return self._make_table(md)

    def get_table(self, path: Path, if_not_exists: IfNotExistsParam) -> Table | None:
        md = self._client.send_request('CatalogBase', 'get_table', {'path': path, 'if_not_exists': if_not_exists})
        return None if md is None else self._make_table(md)

    def get_table_by_id(
        self, tbl_id: UUID, version: int | None = None, ignore_if_dropped: bool = False
    ) -> Table | None:
        md = self._client.send_request(
            'CatalogBase',
            'get_table_by_id',
            {'tbl_id': tbl_id, 'version': version, 'ignore_if_dropped': ignore_if_dropped},
        )
        return None if md is None else self._make_table(md)

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
