from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple
from uuid import UUID

import sqlalchemy as sql

from .expr import Expr
from .data_row import DataRow
from .row_builder import RowBuilder
import pixeltable.iterators as iters
import pixeltable.exceptions as excs
import pixeltable.catalog as catalog


class ColumnRef(Expr):
    """A reference to a table column

    When this reference is created in the context of a view, it can also refer to a column of the view base.
    For that reason, a ColumnRef needs to be serialized with the qualifying table id (column ids are only
    unique in the context of a particular table).
    """
    def __init__(self, col: catalog.Column):
        super().__init__(col.col_type)
        assert col.tbl is not None
        self.col = col
        self.is_unstored_iter_col = \
            col.tbl.is_component_view() and col.tbl.is_iterator_column(col) and not col.is_stored
        self.iter_arg_ctx: Optional[RowBuilder.EvalCtx] = None
        # number of rowid columns in the base table
        self.base_rowid_len = col.tbl.base.num_rowid_columns() if self.is_unstored_iter_col else 0
        self.base_rowid = [None] * self.base_rowid_len
        self.iterator: Optional[iters.ComponentIterator] = None
        # index of the position column in the view's primary key; don't try to reference tbl.store_tbl here
        self.pos_idx: Optional[int] = col.tbl.num_rowid_columns() - 1 if self.is_unstored_iter_col else None
        self.id = self._create_id()

    def set_iter_arg_ctx(self, iter_arg_ctx: RowBuilder.EvalCtx) -> None:
        self.iter_arg_ctx = iter_arg_ctx
        assert len(self.iter_arg_ctx.target_slot_idxs) == 1  # a single inline dict

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('tbl_id', self.col.tbl.id), ('col_id', self.col.id)]

    def __getattr__(self, name: str) -> Expr:
        from .column_property_ref import ColumnPropertyRef
        # resolve column properties
        if name == ColumnPropertyRef.Property.ERRORTYPE.name.lower() \
                or name == ColumnPropertyRef.Property.ERRORMSG.name.lower():
            if not (self.col.is_computed and self.col.is_stored) and not self.col.col_type.is_media_type():
                raise excs.Error(f'{name} only valid for a stored computed or media column: {self}')
            return ColumnPropertyRef(self, ColumnPropertyRef.Property[name.upper()])
        if name == ColumnPropertyRef.Property.FILEURL.name.lower() \
                or name == ColumnPropertyRef.Property.LOCALPATH.name.lower():
            if not self.col.col_type.is_media_type():
                raise excs.Error(f'{name} only valid for image/video/audio/document columns: {self}')
            if self.col.is_computed and not self.col.is_stored:
                raise excs.Error(f'{name} not valid for computed unstored columns: {self}')
            return ColumnPropertyRef(self, ColumnPropertyRef.Property[name.upper()])

        if self.col_type.is_json_type():
            from .json_path import JsonPath
            return JsonPath(self, [name])

        return super().__getattr__(name)

    def default_column_name(self) -> Optional[str]:
        return str(self)

    def _equals(self, other: ColumnRef) -> bool:
        return self.col == other.col

    def __str__(self) -> str:
        return self.col.name

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return self.col.sa_col

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        if not self.is_unstored_iter_col:
            assert data_row.has_val[self.slot_idx]
            return

        # if this is a new base row, we need to instantiate a new iterator
        if self.base_rowid != data_row.pk[:self.base_rowid_len]:
            row_builder.eval(data_row, self.iter_arg_ctx)
            iterator_args = data_row[self.iter_arg_ctx.target_slot_idxs[0]]
            self.iterator = self.col.tbl.iterator_cls(**iterator_args)
            self.base_rowid = data_row.pk[:self.base_rowid_len]
        self.iterator.set_pos(data_row.pk[self.pos_idx])
        res = next(self.iterator)
        data_row[self.slot_idx] = res[self.col.name]

    def _as_dict(self) -> Dict:
        tbl = self.col.tbl
        version = tbl.version if tbl.is_snapshot else None
        return {'tbl_id': str(tbl.id), 'tbl_version': version, 'col_id': self.col.id}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        tbl_id, version, col_id = UUID(d['tbl_id']), d['tbl_version'], d['col_id']
        tbl_version = catalog.Catalog.get().tbl_versions[(tbl_id, version)]
        assert col_id in tbl_version.cols_by_id
        col = tbl_version.cols_by_id[col_id]
        return cls(col)

