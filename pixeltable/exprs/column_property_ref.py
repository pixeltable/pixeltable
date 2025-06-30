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
        CELLMD = 4  # JSON metadata for the cell, e.g. errortype, errormsg for media columns

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

    def is_cellmd_prop(self) -> bool:
        return self.prop in (self.Property.ERRORTYPE, self.Property.ERRORMSG, self.Property.CELLMD)

    def sql_expr(self, sql_elements: SqlElementCache) -> Optional[sql.ColumnElement]:
        if not self._col_ref.col_handle.get().is_stored:
            return None
        col = self._col_ref.col_handle.get()

        # the errortype/-msg properties of a read-validated media column need to be extracted from the DataRow
        if (
            col.col_type.is_media_type()
            and col.media_validation == catalog.MediaValidation.ON_READ
            and self.is_cellmd_prop()
        ):
            return None

        if self.prop == self.Property.ERRORTYPE:
            return col.sa_cellmd_col.op('->>')('errortype')
            return col.sa_cellmd_col['errortype']
            # raise NotImplementedError  # TODO jgp Access errortype property of JSON field in datarow
        if self.prop == self.Property.ERRORMSG:
            return col.sa_cellmd_col.op('->>')('errormsg')
            return col.sa_cellmd_col['errormsg']
            # raise NotImplementedError  # TODO jgp Access errormsg property of JSON field in datarow
        if self.prop == self.Property.CELLMD:
            assert col.sa_cellmd_col is not None
            return col.sa_cellmd_col
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
        elif self.is_cellmd_prop():
            exc = data_row.get_exc(self._col_ref.slot_idx)
            if exc is None:
                data_row[self.slot_idx] = None
            elif self.prop == self.Property.ERRORTYPE:
                data_row[self.slot_idx] = type(exc).__name__
            elif self.prop == self.Property.ERRORMSG:
                data_row[self.slot_idx] = str(exc)
            else:
                # Create JSON from an exception for the CELLMD property.
                data_row[self.slot_idx] = {'errortype': type(exc).__name__, 'errormsg': str(exc)}
                # raise NotImplementedError  # TODO jgp Create a cellmd when to store exc info (Construct JSON)
            return
        else:
            raise AssertionError()

    def _as_dict(self) -> dict:
        return {'prop': self.prop.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> ColumnPropertyRef:
        assert 'prop' in d
        assert isinstance(components[0], ColumnRef)
        return cls(components[0], cls.Property(d['prop']))
