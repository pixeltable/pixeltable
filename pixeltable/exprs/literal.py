from __future__ import annotations

import datetime
from typing import Any, Optional

import numpy as np
import sqlalchemy as sql

import pixeltable.type_system as ts
from pixeltable.env import Env

from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class Literal(Expr):
    val: Any

    def __init__(self, val: Any, col_type: Optional[ts.ColumnType] = None):
        if col_type is not None:
            val = col_type.create_literal(val)
        else:
            # try to determine a type for val
            col_type = ts.ColumnType.infer_literal_type(val)
            if col_type is None:
                raise TypeError(f'Not a valid literal: {val}')
        super().__init__(col_type)
        if isinstance(val, datetime.datetime):
            # Normalize the datetime to UTC: all timestamps are stored as UTC (both in the database and in literals)
            if val.tzinfo is None:
                # We have a naive datetime. Modify it to use the configured default time zone
                default_tz = Env.get().default_time_zone
                if default_tz is not None:
                    val = val.replace(tzinfo=default_tz)
            # Now convert to UTC
            val = val.astimezone(datetime.timezone.utc)
        if isinstance(val, tuple):
            # Tuples are stored as a list
            val = list(val)
        self.val = val
        self.id = self._create_id()

    def default_column_name(self) -> Optional[str]:
        return 'Literal'

    def __str__(self) -> str:
        if self.col_type.is_string_type():
            return f"'{self.val}'"
        if self.col_type.is_timestamp_type():
            assert isinstance(self.val, datetime.datetime)
            default_tz = Env.get().default_time_zone
            return f"'{self.val.astimezone(default_tz).isoformat()}'"
        if self.col_type.is_date_type():
            assert isinstance(self.val, datetime.date)
            return f"'{self.val.isoformat()}'"
        if self.col_type.is_array_type():
            assert isinstance(self.val, np.ndarray)
            return str(self.val.tolist())
        return str(self.val)

    def __repr__(self) -> str:
        return f'Literal({self.val!r})'

    def _equals(self, other: Literal) -> bool:
        return self.val == other.val

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('val', self.val)]

    def sql_expr(self, _: SqlElementCache) -> Optional[sql.ColumnElement]:
        # Return a sql object so that constants can participate in SQL expressions
        return sql.sql.expression.literal(self.val, type_=self.col_type.to_sa_type())

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        # this will be called, even though sql_expr() does not return None
        data_row[self.slot_idx] = self.val

    def _as_dict(self) -> dict:
        # For some types, we need to explicitly record their type, because JSON does not know
        # how to interpret them unambiguously
        if self.col_type.is_timestamp_type():
            assert isinstance(self.val, datetime.datetime)
            assert self.val.tzinfo == datetime.timezone.utc  # Must be UTC in a literal
            # Convert to ISO format in UTC (in keeping with the principle: all timestamps are
            # stored as UTC in the database)
            encoded_val = self.val.isoformat()
            return {'val': encoded_val, 'val_t': self.col_type._type.name, **super()._as_dict()}
        elif self.col_type.is_date_type():
            assert isinstance(self.val, datetime.date)
            encoded_val = self.val.isoformat()
            return {'val': encoded_val, 'val_t': self.col_type._type.name, **super()._as_dict()}
        elif self.col_type.is_array_type():
            assert isinstance(self.val, np.ndarray)
            return {'val': self.val.tolist(), 'val_t': self.col_type._type.name, **super()._as_dict()}
        else:
            return {'val': self.val, **super()._as_dict()}

    def as_literal(self) -> Optional[Literal]:
        return self

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> Literal:
        assert 'val' in d
        if 'val_t' in d:
            val_t = d['val_t']
            if val_t == ts.ColumnType.Type.DATE.name:
                dt = datetime.date.fromisoformat(d['val'])
                return cls(dt)
            elif val_t == ts.ColumnType.Type.TIMESTAMP.name:
                dt = datetime.datetime.fromisoformat(d['val'])
                assert dt.tzinfo == datetime.timezone.utc  # Must be UTC in the database
                return cls(dt)
            elif val_t == ts.ColumnType.Type.ARRAY.name:
                arrays = np.array(d['val'])
                return cls(arrays)
        return cls(d['val'])
