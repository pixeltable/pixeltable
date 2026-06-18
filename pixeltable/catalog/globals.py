from __future__ import annotations

import dataclasses
import enum
import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.metadata import schema

if TYPE_CHECKING:
    from pixeltable import exprs
    from pixeltable.types import ColumnSpec

# name of the position column in a component view
_POS_COLUMN_NAME = 'pos'
_ROWID_COLUMN_NAME = '_rowid'

# Set of symbols that are predefined in the `InsertableTable` class (and are therefore not allowed as column names).
# This will be populated lazily to avoid circular imports.
_PREDEF_SYMBOLS: set[str] | None = None


@dataclasses.dataclass
class DirEntry:
    dir: schema.Dir | None
    dir_entries: dict[str, DirEntry]
    table: schema.Table | None

    # Only populated for table entries when get_dir_contents() was called with with_error_counts=True;
    # None otherwise (including for directory entries).
    table_error_count: int | None = None


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
        return dataclasses.asdict(self, dict_factory=schema.md_dict_factory)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableVersionMd:
        return schema.md_from_dict(cls, data)


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
class QColumnId:
    """Qualified column id"""

    tbl_id: UUID
    col_id: int


class MediaValidation(enum.Enum):
    ON_READ = 0
    ON_WRITE = 1

    @classmethod
    def validated(cls, name: str, error_prefix: str) -> MediaValidation:
        try:
            return cls[name.upper()]
        except KeyError:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, f'{error_prefix} must be one of: [{val_strs}]'
            ) from None


class IfExistsParam(enum.Enum):
    ERROR = 0
    IGNORE = 1
    REPLACE = 2
    REPLACE_FORCE = 3

    @classmethod
    def validated(cls, param_val: str, param_name: str) -> IfExistsParam:
        try:
            return cls[param_val.upper()]
        except KeyError:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, f'{param_name} must be one of: [{val_strs}]'
            ) from None


class IfNotExistsParam(enum.Enum):
    ERROR = 0
    IGNORE = 1

    @classmethod
    def validated(cls, param_val: str, param_name: str) -> IfNotExistsParam:
        try:
            return cls[param_val.upper()]
        except KeyError:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, f'{param_name} must be one of: [{val_strs}]'
            ) from None


def is_valid_identifier(name: str, *, allow_hyphens: bool = False) -> bool:
    # If allow_hyphens=True, we allow hyphens to appear in the name, but we still do not permit a name to start with one
    adj_name = name.replace('-', '_') if allow_hyphens else name
    return adj_name.isidentifier() and not name.startswith('-') and not name.startswith('_')


def is_system_column_name(name: str) -> bool:
    from pixeltable.catalog import InsertableTable, View

    global _PREDEF_SYMBOLS  # noqa: PLW0603
    if _PREDEF_SYMBOLS is None:
        _PREDEF_SYMBOLS = set(itertools.chain(dir(InsertableTable), dir(View)))
    return name in _PREDEF_SYMBOLS


def normalize_schema(schema: Mapping[str, type | ColumnSpec | exprs.Expr]) -> dict[str, ColumnSpec]:
    """Canonicalize a create_table schema to a {name: ColumnSpec} mapping with resolved ColumnTypes."""
    from pixeltable import exprs

    result: dict[str, ColumnSpec] = {}
    for name, spec in schema.items():
        if isinstance(spec, exprs.Expr):
            result[name] = {'value': spec}
            continue
        col_spec: dict[str, Any] = dict(spec) if isinstance(spec, dict) else {'type': spec}
        if col_spec.get('type') is not None:
            col_spec['type'] = ts.ColumnType.normalize_type(
                col_spec['type'], nullable_default=True, allow_builtin_types=False
            )
        result[name] = cast('ColumnSpec', col_spec)
    return result
