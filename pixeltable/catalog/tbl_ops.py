# This file contains all dataclasses related to schema.PendingTableOp:
# - TableOp: the container for each log entry
# - <>Op: the actual operation; each <>Op class contains enough information to exec/undo itself given a
#   TableVersion instance

from __future__ import annotations

import dataclasses
import logging
import sys
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import sqlalchemy as sql

import pixeltable.metadata.schema as schema
from pixeltable.runtime import get_runtime

from .update_status import UpdateStatus

if TYPE_CHECKING:
    from pixeltable.catalog.table_version import TableVersion

_logger = logging.getLogger('pixeltable')


class OpStatus(Enum):
    PENDING = 0
    COMPLETED = 1
    ABORTED = 2


@dataclasses.dataclass
class TableOp:
    """TableOp describes an individual operation that needs to be performed on the table.

    If needs_xact is True, the TableOp is executed, and its state is updated as part of a single store transaction.
    Otherwise, the op is executed outside of the store transaction. Such operations (including undo) must be idempotent
    and safe to execute concurrently, because multiple processes may attempt to make progress on the same TableOp
    simultaneously.
    """

    needs_tv: ClassVar[bool]  # if False, exec/undo can be called with tv=None
    needs_xact: ClassVar[bool]  # whether this op must run as part of a transaction

    tbl_id: str  # uuid.UUID
    op_sn: int  # sequence number within the update operation; [0, num_ops)
    num_ops: int  # total number of ops forming the update operation
    status: OpStatus

    def to_dict(self) -> dict:
        result = dataclasses.asdict(self, dict_factory=schema.md_dict_factory)
        result['_classname'] = self.__class__.__name__
        return result

    @classmethod
    def from_dict(cls, data: dict) -> TableOp:
        classname = data.pop('_classname')
        # needs_xact used to be a member variable. Remove it from the dict for backward compatibility.
        # TODO: delete this line, and the assert the follows, in ~ May 2026 or later. The chance of anyone still having
        # needs_xact in their pending table ops at that point will be extremely low.
        needs_xact_legacy = data.pop('needs_xact', None)
        op_class = getattr(sys.modules[__name__], classname)
        op = schema.md_from_dict(op_class, data)
        if needs_xact_legacy is not None:
            assert op.needs_xact == needs_xact_legacy
        return op

    def exec(self, tv: TableVersion | None) -> None:
        raise NotImplementedError(f'{self.__class__.__name__}.exec()')

    def undo(self, tv: TableVersion | None) -> None:
        raise NotImplementedError(f'{self.__class__.__name__}.undo()')


@dataclasses.dataclass
class CreateStoreTableOp(TableOp):
    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = False

    def exec(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        with get_runtime().begin_xact():
            tv.store_tbl.create()

    def undo(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        with get_runtime().begin_xact():
            tv.store_tbl.drop()


@dataclasses.dataclass
class CreateStoreIdxsOp(TableOp):
    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = False

    idx_ids: list[int]

    def exec(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        for idx_id in self.idx_ids:
            with get_runtime().begin_xact():
                tv.store_tbl.create_index(idx_id)

    def undo(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        for idx_id in self.idx_ids:
            with get_runtime().begin_xact():
                tv.store_tbl.drop_index(idx_id)


@dataclasses.dataclass
class LoadViewOp(TableOp):
    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = True

    view_path: dict[str, Any]  # needed to create the view load plan

    def exec(self, tv: TableVersion | None) -> None:
        from pixeltable.catalog.table_version_path import TableVersionPath
        from pixeltable.plan import Planner

        assert get_runtime().in_xact
        view_path = TableVersionPath.from_dict(self.view_path)
        plan, _ = Planner.create_view_load_plan(view_path)
        with get_runtime().report_progress():
            plan.ctx.title = tv.display_str()
            _, row_counts = tv.store_tbl.insert_rows(plan, v_min=tv.version)
        status = UpdateStatus(row_count_stats=row_counts)
        get_runtime().catalog.store_update_status(tv.id, tv.version, status)
        _logger.debug(f'Loaded view {tv.name} with {row_counts.num_rows} rows')

    def undo(self, tv: TableVersion | None) -> None:
        from pixeltable.utils.filecache import FileCache

        # clear out any media files
        tv.delete_media()
        FileCache.get().clear(tbl_id=tv.id)


@dataclasses.dataclass
class CreateTableMdOp(TableOp):
    """Undo-only log record"""

    needs_tv: ClassVar[bool] = False
    needs_xact: ClassVar[bool] = True

    def exec(self, tv: TableVersion | None) -> None:
        pass

    def undo(self, tv: TableVersion | None) -> None:
        assert get_runtime().in_xact
        get_runtime().catalog.delete_tbl_md(uuid.UUID(self.tbl_id))


@dataclasses.dataclass
class DeleteTableMdOp(TableOp):
    needs_tv: ClassVar[bool] = False
    needs_xact: ClassVar[bool] = True

    def exec(self, tv: TableVersion | None) -> None:
        assert get_runtime().in_xact
        get_runtime().catalog.delete_tbl_md(uuid.UUID(self.tbl_id))

    def undo(self, tv: TableVersion | None) -> None:
        raise AssertionError()


@dataclasses.dataclass
class CreateTableVersionOp(TableOp):
    """Undo-only log record"""

    needs_tv: ClassVar[bool] = False
    needs_xact: ClassVar[bool] = True

    def exec(self, tv: TableVersion | None) -> None:
        pass

    def undo(self, tv: TableVersion | None) -> None:
        assert get_runtime().in_xact
        get_runtime().catalog.delete_current_tbl_version_md(uuid.UUID(self.tbl_id))


@dataclasses.dataclass
class CreateColumnMdOp(TableOp):
    """Undo-only log record"""

    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = True

    column_ids: list[int]

    def exec(self, tv: TableVersion | None) -> None:
        pass

    def undo(self, tv: TableVersion | None) -> None:
        # TODO this is completely broken, but the fix requires a separate change and more thought. Leaving as is for now
        # because this change is meant to be mostly a refactoring (and a minor change in behavior, but elsewhere)
        # 1. major: write_tbl_md cannot be called while there are pending ops (and we are inside one of them).
        # 2. minor: [] is not an acceptable value for pending_ops
        # 3. minor: TableVersion internals access. Once we figure out how to fix 1, this one should go away as well.
        assert get_runtime().in_xact
        for col_id in self.column_ids:
            del tv._tbl_md.column_md[col_id]
        get_runtime().catalog.write_tbl_md(tv.id, None, tv._tbl_md, None, None, [])


@dataclasses.dataclass
class CreateStoreColumnsOp(TableOp):
    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = False

    column_ids: list[int]

    def exec(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        for col_id in self.column_ids:
            with get_runtime().begin_xact():
                tv.store_tbl.add_column(tv.cols_by_id[col_id], if_not_exists=True)

    def undo(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        for col_id in self.column_ids:
            with get_runtime().begin_xact():
                tv.store_tbl.drop_column(tv.cols_by_id[col_id], if_exists=True)


@dataclasses.dataclass
class SetColumnValueOp(TableOp):
    """Set values for specified columns."""

    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = False

    column_ids: list[int]

    def exec(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        cols = [tv.cols_by_id[col_id] for col_id in self.column_ids]
        with get_runtime().begin_xact():
            tv._populate_default_values(cols)

    def undo(self, tv: TableVersion | None) -> None:
        pass


@dataclasses.dataclass
class DeleteTableMediaFilesOp(TableOp):
    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = False

    def exec(self, tv: TableVersion | None) -> None:
        from pixeltable.utils.filecache import FileCache

        tv.delete_media()
        FileCache.get().clear(tbl_id=tv.id)

    def undo(self, tv: TableVersion | None) -> None:
        raise AssertionError()


@dataclasses.dataclass
class DropStoreTableOp(TableOp):
    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = False

    def exec(self, tv: TableVersion | None) -> None:
        from pixeltable.store import StoreBase

        # don't reference tv.store_tbl here, it needs to reference the metadata for our base table, which at
        # this point may not exist anymore
        assert not get_runtime().in_xact
        with get_runtime().begin_xact() as conn:
            drop_stmt = f'DROP TABLE IF EXISTS {StoreBase.storage_name(tv.id, tv.is_view)}'
            conn.execute(sql.text(drop_stmt))

    def undo(self, tv: TableVersion | None) -> None:
        raise AssertionError()
