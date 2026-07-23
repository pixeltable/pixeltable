from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from uuid import UUID

    from pixeltable import exprs, func
    from pixeltable._query import Query
    from pixeltable.plan import SampleClause
    from pixeltable.types import ColumnSpec

    from .dir import Dir
    from .globals import DirEntry, IfExistsParam, IfNotExistsParam, MediaValidation
    from .model import IndexSpec
    from .path import Path
    from .table import Table
    from .table_path import TablePath


class CatalogBase(abc.ABC):
    """
    The public catalog API.
    """

    @abc.abstractmethod
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
    ) -> tuple[Table, bool]: ...

    @abc.abstractmethod
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
    ) -> tuple[Table, bool]: ...

    @abc.abstractmethod
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
        idxs: dict[str, 'IndexSpec'],
    ) -> tuple[Table, bool]: ...

    @abc.abstractmethod
    def get_table(self, path: Path, if_not_exists: IfNotExistsParam) -> Table | None: ...

    @abc.abstractmethod
    def get_table_by_id(
        self, tbl_id: UUID, version: int | None = None, ignore_if_dropped: bool = False
    ) -> Table | None: ...

    @abc.abstractmethod
    def drop_table(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None: ...

    @abc.abstractmethod
    def move(self, path: Path, new_path: Path, if_exists: IfExistsParam, if_not_exists: IfNotExistsParam) -> None: ...

    @abc.abstractmethod
    def get_dir_contents(
        self, dir_path: Path, recursive: bool = False, with_error_counts: bool = False
    ) -> dict[str, DirEntry]: ...

    @abc.abstractmethod
    def create_dir(self, path: Path, if_exists: IfExistsParam, parents: bool) -> Dir: ...

    @abc.abstractmethod
    def drop_dir(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None: ...
