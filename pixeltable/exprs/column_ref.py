from __future__ import annotations
from typing import Optional, Any, Tuple
from uuid import UUID

import sqlalchemy as sql

from .expr import Expr
from .data_row import DataRow
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache
import pixeltable.iterators as iters
import pixeltable.exceptions as excs
import pixeltable.catalog as catalog


class ColumnRef(Expr):
    """A reference to a table column

    When this reference is created in the context of a view, it can also refer to a column of the view base.
    For that reason, a ColumnRef needs to be serialized with the qualifying table id (column ids are only
    unique in the context of a particular table).

    Media validation:
    - media validation is potentially cpu-intensive, and it's desirable to schedule and parallelize it during
      general expr evaluation
    - media validation on read is done in ColumnRef.eval()
    - a validating ColumnRef cannot be translated to SQL (because the validation is done in Python)
    - in that case, the ColumnRef also instantiates a second non-validating ColumnRef as a component (= dependency)
    - the non-validating ColumnRef is used for SQL translation
    """

    col: catalog.Column
    is_unstored_iter_col: bool
    iter_arg_ctx: Optional[RowBuilder.EvalCtx]
    base_rowid_len: int
    base_rowid: list[Optional[Any]]
    iterator: Optional[iters.ComponentIterator]
    pos_idx: Optional[int]
    id: int
    perform_validation: bool  # if True, performs media validation

    def __init__(self, col: catalog.Column, perform_validation: Optional[bool] = None):
        super().__init__(col.col_type)
        assert col.tbl is not None
        self.col = col
        self.is_unstored_iter_col = \
            col.tbl.is_component_view() and col.tbl.is_iterator_column(col) and not col.is_stored
        self.iter_arg_ctx = None
        # number of rowid columns in the base table
        self.base_rowid_len = col.tbl.base.num_rowid_columns() if self.is_unstored_iter_col else 0
        self.base_rowid = [None] * self.base_rowid_len
        self.iterator = None
        # index of the position column in the view's primary key; don't try to reference tbl.store_tbl here
        self.pos_idx = col.tbl.num_rowid_columns() - 1 if self.is_unstored_iter_col else None

        self.perform_validation = False
        if col.col_type.is_media_type():
            # we perform media validation if the column is a media type and the validation is set to ON_READ,
            # unless we're told not to
            self.perform_validation = perform_validation if perform_validation is not None else (
                col.col_type.is_media_type() and col.media_validation == catalog.MediaValidation.ON_READ
            )
        if self.perform_validation:
            non_validating_col_ref = ColumnRef(col, perform_validation=False)
            self.components = [non_validating_col_ref]
        self.id = self._create_id()

    def set_iter_arg_ctx(self, iter_arg_ctx: RowBuilder.EvalCtx) -> None:
        self.iter_arg_ctx = iter_arg_ctx
        assert len(self.iter_arg_ctx.target_slot_idxs) == 1  # a single inline dict

    def _id_attrs(self) -> list[Tuple[str, Any]]:
        return (
            super()._id_attrs()
            + [('tbl_id', self.col.tbl.id), ('col_id', self.col.id), ('validates', self.perform_validation)]
        )

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

    def similarity(self, item: Any, *, idx: Optional[str] = None) -> Expr:
        from .similarity_expr import SimilarityExpr
        return SimilarityExpr(self, item, idx_name=idx)

    def default_column_name(self) -> Optional[str]:
        return str(self)

    def _equals(self, other: ColumnRef) -> bool:
        return self.col == other.col and self.perform_validation == other.perform_validation

    def __str__(self) -> str:
        if self.col.name is None:
            return f'<unnamed column {self.col.id}>'
        else:
            return self.col.name

    def __repr__(self) -> str:
        return f'ColumnRef({self.col!r})'

    def sql_expr(self, _: SqlElementCache) -> Optional[sql.ColumnElement]:
        return None if self.perform_validation else self.col.sa_col

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        if self.perform_validation:
            # validate media file of our input ColumnRef and if successful, replicate the state of that slot
            # to our slot
            unvalidated_slot_idx = self.components[0].slot_idx
            if data_row.file_paths[unvalidated_slot_idx] is None:
                # no media file to validate, we still need to replicate the value
                assert data_row.file_urls[unvalidated_slot_idx] is None
                val = data_row.vals[unvalidated_slot_idx]
                data_row.vals[self.slot_idx] = val
                data_row.has_val[self.slot_idx] = True
                return

            try:
                self.col.col_type.validate_media(data_row.file_paths[unvalidated_slot_idx])
                # access the value only after successful validation
                val = data_row[unvalidated_slot_idx]
                data_row.vals[self.slot_idx] = val
                data_row.has_val[self.slot_idx] = True
                # make sure that the validated slot points to the same file as the unvalidated slot
                data_row.file_paths[self.slot_idx] = data_row.file_paths[unvalidated_slot_idx]
                data_row.file_urls[self.slot_idx] = data_row.file_urls[unvalidated_slot_idx]
                return
            except excs.Error as exc:
                # propagate the exception, but ignore it otherwise;
                # media validation errors don't cause exceptions during query execution
                # TODO: allow for different error-handling behavior
                row_builder.set_exc(data_row, self.slot_idx, exc)
                return

        if not self.is_unstored_iter_col:
            # supply default
            data_row[self.slot_idx] = None
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

    def _as_dict(self) -> dict:
        tbl = self.col.tbl
        version = tbl.version if tbl.is_snapshot else None
        return {'tbl_id': str(tbl.id), 'tbl_version': version, 'col_id': self.col.id}

    @classmethod
    def get_column(cls, d: dict) -> catalog.Column:
        tbl_id, version, col_id = UUID(d['tbl_id']), d['tbl_version'], d['col_id']
        tbl_version = catalog.Catalog.get().tbl_versions[(tbl_id, version)]
        # don't use tbl_version.cols_by_id here, this might be a snapshot reference to a column that was then dropped
        col = next(col for col in tbl_version.cols if col.id == col_id)
        return col

    @classmethod
    def _from_dict(cls, d: dict, _: list[Expr]) -> Expr:
        col = cls.get_column(d)
        return cls(col)
