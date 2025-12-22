# This file contains all dataclasses related to schema.PendingTableOp:
# - TableOp: the container for each log entry
# - <>Op: the actual operation, which is performed by TableVersion.exec_op(); each <>Op class contains
#   enough information for exec_op() to perform the operation without having to reference data outside of
#   TableVersion

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

import pixeltable.metadata.schema as schema


class OpStatus(Enum):
    PENDING = 0
    COMPLETED = 1
    ABORTED = 2


@dataclasses.dataclass
class TableOp:
    tbl_id: str  # uuid.UUID
    op_sn: int  # sequence number within the update operation; [0, num_ops)
    num_ops: int  # total number of ops forming the update operation
    needs_xact: bool  # if True, op must be run as part of a transaction
    status: OpStatus

    def to_dict(self) -> dict:
        result = dataclasses.asdict(self, dict_factory=schema.md_dict_factory)
        result['_classname'] = self.__class__.__name__
        return result

    @classmethod
    def from_dict(cls, data: dict) -> TableOp:
        import pixeltable.catalog.tbl_ops as tbl_ops

        classname = data.pop('_classname')
        op_class = getattr(tbl_ops, classname)
        return schema.md_from_dict(op_class, data)


@dataclasses.dataclass
class CreateStoreTableOp(TableOp):
    pass


@dataclasses.dataclass
class CreateStoreIdxsOp(TableOp):
    idx_ids: list[int]


@dataclasses.dataclass
class LoadViewOp(TableOp):
    view_path: dict[str, Any]  # needed to create the view load plan


@dataclasses.dataclass
class CreateTableMdOp(TableOp):
    """Undo-only log record"""

    pass


@dataclasses.dataclass
class DeleteTableMdOp(TableOp):
    pass


@dataclasses.dataclass
class CreateTableVersionOp(TableOp):
    """Undo-only log record"""

    preceding_version: int
    preceding_schema_version: int | None  # only set for schema changes


@dataclasses.dataclass
class CreateColumnMdOp(TableOp):
    """Undo-only log record"""

    column_ids: list[int]


@dataclasses.dataclass
class CreateStoreColumnsOp(TableOp):
    column_ids: list[int]


@dataclasses.dataclass
class DeleteTableMediaFilesOp(TableOp):
    pass


@dataclasses.dataclass
class DropStoreTableOp(TableOp):
    pass
