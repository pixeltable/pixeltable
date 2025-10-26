# This file contains all dataclasses related to schema.PendingTableOp:
# - TableOp: the container for each log entry
# - <>Op: the actual operation, which is performed by TableVersion.exec_op(); each <>Op class contains
#   enough information for exec_op() to perform the operation without having to reference data outside of
#   TableVersion

import dataclasses
from typing import Any


@dataclasses.dataclass
class CreateStoreTableOp:
    pass


@dataclasses.dataclass
class CreateIndexOp:
    idx_id: int


@dataclasses.dataclass
class LoadViewOp:
    view_path: dict[str, Any]  # needed to create the view load plan


@dataclasses.dataclass
class DeleteTableMdOp:
    pass


@dataclasses.dataclass
class DeleteTableMediaFilesOp:
    pass


@dataclasses.dataclass
class DropStoreTableOp:
    pass


@dataclasses.dataclass
class TableOp:
    tbl_id: str  # uuid.UUID
    op_sn: int  # sequence number within the update operation; [0, num_ops)
    num_ops: int  # total number of ops forming the update operation
    needs_xact: bool  # if True, op must be run as part of a transaction

    create_store_table_op: CreateStoreTableOp | None = None
    create_index_op: CreateIndexOp | None = None
    load_view_op: LoadViewOp | None = None
