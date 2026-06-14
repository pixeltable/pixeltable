from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from .catalog_base import CatalogBase
from .insertable_table_proxy import InsertableTableProxy

if TYPE_CHECKING:
    from uuid import UUID

    from pixeltable import exprs, func
    from pixeltable.plan import SampleClause
    from pixeltable.types import ColumnSpec

    from .dir import Dir
    from .globals import DirEntry, IfExistsParam, IfNotExistsParam, MediaValidation
    from .path import Path
    from .table import Table
    from .table_path import TablePath


class CatalogProxy(CatalogBase):
    """
    An implementation of CatalogBase that delegates to a hosted catalog .
    """

    def __init__(self, catalog_uri: str):
        self._catalog_uri = catalog_uri

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
        return InsertableTableProxy(path), True

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
        raise NotImplementedError

    def get_table(self, path: Path, if_not_exists: IfNotExistsParam) -> Table | None:
        raise NotImplementedError

    def get_table_by_id(
        self, tbl_id: UUID, version: int | None = None, ignore_if_dropped: bool = False
    ) -> Table | None:
        raise NotImplementedError

    def drop_table(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None:
        raise NotImplementedError

    def move(self, path: Path, new_path: Path, if_exists: IfExistsParam, if_not_exists: IfNotExistsParam) -> None:
        raise NotImplementedError

    def get_dir_contents(
        self, dir_path: Path, recursive: bool = False, with_error_counts: bool = False
    ) -> dict[str, DirEntry]:
        raise NotImplementedError

    def create_dir(self, path: Path, if_exists: IfExistsParam, parents: bool) -> Dir:
        raise NotImplementedError

    def drop_dir(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None:
        raise NotImplementedError
