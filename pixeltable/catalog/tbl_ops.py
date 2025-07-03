import dataclasses
from typing import Any, Optional


@dataclasses.dataclass
class CreateStoreTableOp:
    pass


@dataclasses.dataclass
class LoadViewOp:
    view_path: dict[str, Any]


@dataclasses.dataclass
class TableOp:
    tbl_id: str  # uuid.UUID
    seq_num: int
    needs_xact: bool  # if True, op must be run as part of a transaction

    create_store_table_op: Optional[CreateStoreTableOp] = None
    load_view_op: Optional[LoadViewOp] = None
