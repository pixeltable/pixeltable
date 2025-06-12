from __future__ import annotations

import enum
from typing import Any, Optional

import sqlalchemy as sql

import pixeltable.type_system as ts
from pixeltable import catalog

from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class ColumnPropertyRef(Expr):
    """A reference to a property of a table column

    The properties themselves are type-specific and may or may not need to reference the underlying column data.
    """

    class Property(enum.Enum):
        ERRORTYPE = 0
        ERRORMSG = 1
        FILEURL = 2
        LOCALPATH = 3

    def __init__(self, col_ref: ColumnRef, prop: Property):
        super().__init__(ts.StringType(nullable=True))
        self.components = [col_ref]
        self.prop = prop
        self.id = self._create_id()

    def default_column_name(self) -> Optional[str]:
        return str(self).replace('.', '_')

    def _equals(self, other: ColumnPropertyRef) -> bool:
        return self.prop == other.prop

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('prop', self.prop.value)]

    @property
    def _col_ref(self) -> ColumnRef:
        col_ref = self.components[0]
        assert isinstance(col_ref, ColumnRef)
        return col_ref

    def __repr__(self) -> str:
        return f'{self._col_ref}.{self.prop.name.lower()}'

    def is_error_prop(self) -> bool:
        return self.prop in (self.Property.ERRORTYPE, self.Property.ERRORMSG)

    def sql_expr(self, sql_elements: SqlElementCache) -> Optional[sql.ColumnElement]:
        if not self._col_ref.col.is_stored:
            return None

        # we need to reestablish that we have the correct Column instance, there could have been a metadata
        # reload since init()
        # TODO: add an explicit prepare phase (ie, Expr.prepare()) that gives every subclass instance a chance to
        # perform runtime checks and update state
        tv = self._col_ref.tbl_version.get()
        assert tv.is_validated
        # we can assume at this point during query execution that the column exists
        assert self._col_ref.col_id in tv.cols_by_id
        col = tv.cols_by_id[self._col_ref.col_id]

        # the errortype/-msg properties of a read-validated media column need to be extracted from the DataRow
        if (
            col.col_type.is_media_type()
            and col.media_validation == catalog.MediaValidation.ON_READ
            and self.is_error_prop()
        ):
            return None

        if self.prop == self.Property.ERRORTYPE:
            assert col.sa_errortype_col is not None
            return col.sa_errortype_col
        if self.prop == self.Property.ERRORMSG:
            assert col.sa_errormsg_col is not None
            return col.sa_errormsg_col
        if self.prop == self.Property.FILEURL:
            # the file url is stored as the column value
            return sql_elements.get(self._col_ref)
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        if self.prop == self.Property.FILEURL:
            assert data_row.has_val[self._col_ref.slot_idx]
            data_row[self.slot_idx] = data_row.file_urls[self._col_ref.slot_idx]
            return
        elif self.prop == self.Property.LOCALPATH:
            assert data_row.has_val[self._col_ref.slot_idx]
            data_row[self.slot_idx] = data_row.file_paths[self._col_ref.slot_idx]
            return
        elif self.is_error_prop():
            exc = data_row.get_exc(self._col_ref.slot_idx)
            if exc is None:
                data_row[self.slot_idx] = None
            elif self.prop == self.Property.ERRORTYPE:
                data_row[self.slot_idx] = type(exc).__name__
            else:
                data_row[self.slot_idx] = str(exc)
        else:
            raise AssertionError()

    def _as_dict(self) -> dict:
        return {'prop': self.prop.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> ColumnPropertyRef:
        assert 'prop' in d
        assert isinstance(components[0], ColumnRef)
        return cls(components[0], cls.Property(d['prop']))
