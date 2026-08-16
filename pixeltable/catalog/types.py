"""Pure data types describing a table and its columns.

These carry no reference to a catalog object, so any module can depend on them.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from uuid import UUID

import pixeltable.type_system as ts
from pixeltable.metadata import schema

from .globals import MediaValidation, QColumnId


@dataclasses.dataclass(frozen=True)
class ColumnVersionMd:
    """
    Complete set of metadata records for a table column.
    """

    # path-context table (the logical owner of this column)
    tbl_id: UUID

    # path-context table's effective version
    effective_version: int | None

    # physical column identify
    qcolid: QColumnId

    # effective version of qcolid.tbl_id
    col_effective_version: int | None

    # version-independent metadata
    col_md: schema.ColumnMd

    # versioned metadata
    schema_col: schema.SchemaColumn

    is_iterator_col: bool = False

    @property
    def id(self) -> int:
        return self.col_md.id

    @property
    def is_system_col(self) -> bool:
        return self.schema_col.name is None

    @property
    def name(self) -> str | None:
        return self.schema_col.name

    @property
    def comment(self) -> str | None:
        return self.schema_col.comment

    @property
    def col_type(self) -> ts.ColumnType:
        return ts.ColumnType.from_dict(self.schema_col.col_type)

    @property
    def is_pk(self) -> bool:
        return self.schema_col.is_pk

    @property
    def is_computed(self) -> bool:
        return self.schema_col.value_expr is not None

    @property
    def is_stored(self) -> bool:
        return self.col_md.stored

    @property
    def media_validation(self) -> MediaValidation | None:
        if self.schema_col.media_validation is None:
            return None
        return MediaValidation[self.schema_col.media_validation.upper()]

    @property
    def stores_cellmd(self) -> bool:
        return self.col_md.stores_cellmd

    def with_context(self, tbl_id: UUID, effective_version: int | None) -> ColumnVersionMd:
        """Reset the path-context table."""
        if tbl_id == self.tbl_id and effective_version == self.effective_version:
            return self
        return ColumnVersionMd(
            tbl_id=tbl_id,
            effective_version=effective_version,
            qcolid=self.qcolid,
            col_effective_version=self.col_effective_version,
            col_md=self.col_md,
            schema_col=self.schema_col,
            is_iterator_col=self.is_iterator_col,
        )

    def retarget(self, col_effective_version: int | None) -> ColumnVersionMd:
        """Retarget to a specific version of the column's containing table."""
        return ColumnVersionMd(
            tbl_id=self.tbl_id,
            effective_version=self.effective_version,
            qcolid=self.qcolid,
            col_effective_version=col_effective_version,
            col_md=self.col_md,
            schema_col=self.schema_col,
            is_iterator_col=self.is_iterator_col,
        )


@dataclasses.dataclass(frozen=True)
class TableVersionMd:
    """
    Complete set of md records for a specific TableVersion instance.
    """

    tbl_md: schema.TableMd
    version_md: schema.VersionMd
    schema_version_md: schema.SchemaVersionMd

    @property
    def is_pure_snapshot(self) -> bool:
        return (
            self.tbl_md.view_md is not None
            and self.tbl_md.view_md.is_snapshot
            and self.tbl_md.view_md.predicate is None
            and self.tbl_md.view_md.sample_clause is None
            and len(self.schema_version_md.columns) == 0
        )

    def as_dict(self) -> dict:
        return schema.md_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableVersionMd:
        return schema.md_from_dict(cls, data)
