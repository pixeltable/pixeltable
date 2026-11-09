from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .model import TableSchemaChangeSet

if TYPE_CHECKING:
    from uuid import UUID

    from pixeltable import exprs, func
    from pixeltable._query import Query
    from pixeltable.plan import SampleClause
    from pixeltable.types import ColumnSpec

    from .dir import Dir
    from .globals import DirEntry, IfExistsParam, IfNotExistsParam, MediaValidation
    from .model import IndexDefinition
    from .path import Path
    from .table import Table
    from .table_metadata import TableMetadata
    from .table_path import TableMdPath, TablePath
    from .types import TableVersionMd


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
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        has_default_idxs: bool,
        is_data_versioned: bool,
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
        has_default_idxs: bool,
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
        has_default_idxs: bool,
        media_validation: MediaValidation,
        comment: str | None,
        custom_metadata: Any,
        iterator: func.GeneratingFunctionCall | None,
        base: 'Query | None',
        idxs: list['IndexDefinition'],
        is_data_versioned: bool,
        if_exists: IfExistsParam,
    ) -> tuple[Table, bool]: ...

    @abc.abstractmethod
    def update_from_model(self, updates: list[TableSchemaChangeSet]) -> None: ...

    @abc.abstractmethod
    def get_table(self, path: Path, if_not_exists: IfNotExistsParam) -> Table | None: ...

    @abc.abstractmethod
    def get_table_by_id(
        self, tbl_id: UUID, version: int | None = None, ignore_if_dropped: bool = False
    ) -> Table | None: ...

    @abc.abstractmethod
    def get_table_paths_md(self, paths: Sequence[Path]) -> list[tuple[TableMdPath, TableMetadata] | None]:
        """Read the metadata of the tables at `paths` in one transaction, with None for a path without a table."""

    @abc.abstractmethod
    def drop_table(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None: ...

    @abc.abstractmethod
    def move(self, path: Path, new_path: Path, if_exists: IfExistsParam, if_not_exists: IfNotExistsParam) -> None: ...

    @abc.abstractmethod
    def get_dir_contents(
        self, dir_path: Path, recursive: bool = False, with_error_counts: bool = False, with_base_paths: bool = False
    ) -> dict[str, DirEntry]: ...

    @abc.abstractmethod
    def create_dir(self, path: Path, if_exists: IfExistsParam, parents: bool) -> Dir: ...

    @abc.abstractmethod
    def drop_dir(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None: ...

    @abc.abstractmethod
    def create_table_from_md(self, path: Path, md: list[TableVersionMd]) -> Table:
        """Create a base table at path with the metadata and history of another base table, and an empty store table.

        The new table is a copy in progress: it refuses every write except load_store_rows(), finish_copy() and
        drop_table().
        """

    @abc.abstractmethod
    def load_store_rows(
        self, tbl_id: UUID, rows_file: str, new_media: dict[str, str], prev_media: dict[str, str]
    ) -> dict[str, str]:
        """Insert the store rows in the Parquet file rows_file into the copy in progress tbl_id.

        Args:
            new_media: url -> local path of each media file that these rows are the first to reference
            prev_media: url -> stored url of each media file that an earlier call stored and these rows reference

        Returns:
            url -> stored url, for each url in new_media
        """

    @abc.abstractmethod
    def finish_copy(self, tbl_id: UUID, num_rows: int, dest_path: Path, if_exists: IfExistsParam) -> bool:
        """Make the copy in progress tbl_id, which must hold num_rows rows, a live table at dest_path.

        Returns whether the table is at dest_path. It is not when if_exists is 'ignore' and another table is at
        dest_path.
        """
