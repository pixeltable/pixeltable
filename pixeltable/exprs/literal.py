from __future__ import annotations

import base64
import datetime
import io
import uuid
from typing import TYPE_CHECKING, Any

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

    def __init__(self, val: Any, col_type: ts.ColumnType | None = None):
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

    def default_column_name(self) -> str | None:
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
        if self.col_type.is_uuid_type():
            assert isinstance(self.val, uuid.UUID)
            return f"'{self.val}'"
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

    def sql_expr(self, _: SqlElementCache) -> sql.ColumnElement | None:
        # Return a sql object so that constants can participate in SQL expressions
        return sql.sql.expression.literal(self.val, type_=self.col_type.to_sa_type())

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        # DataRow holds in-memory values only (not stored format)
        data_row[self.slot_idx] = self.val

    def _as_dict(self) -> dict:
        if self.col_type.is_timestamp_type():
            assert isinstance(self.val, datetime.datetime)
            assert self.val.tzinfo == datetime.timezone.utc  # Must be UTC in a literal
            # Convert to ISO format in UTC (in keeping with the principle: all timestamps are
            # stored as UTC in the database)
            encoded_val = self.val.isoformat()
        elif self.col_type.is_date_type():
            assert isinstance(self.val, datetime.date)
            encoded_val = self.val.isoformat()
        elif self.col_type.is_uuid_type():
            assert isinstance(self.val, uuid.UUID)
            encoded_val = str(self.val)
        elif self.col_type.is_binary_type():
            assert isinstance(self.val, bytes)
            encoded_val = base64.b64encode(self.val).decode('utf-8')
        elif self.col_type.is_array_type():
            assert isinstance(self.val, np.ndarray)
            encoded_val = self.val.tolist()
        else:
            encoded_val = self.val
        return {'val': encoded_val, 'col_type': self.col_type.as_dict(), **super()._as_dict()}

    def as_literal(self) -> Literal | None:
        return self

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> Literal:
        val = d['val']
        col_type_dict = d['col_type']
        if not isinstance(col_type_dict, dict) or '_classname' not in col_type_dict:
            raise ValueError(f'Unsupported or malformed literal type: {col_type_dict}')
        col_type = ts.ColumnType.from_dict(col_type_dict)

        if col_type.is_date_type():
            dt = datetime.date.fromisoformat(val)
            return cls(dt, col_type)
        if col_type.is_timestamp_type():
            dt = datetime.datetime.fromisoformat(val)
            assert dt.tzinfo == datetime.timezone.utc  # Must be UTC in the database
            return cls(dt, col_type)
        if col_type.is_uuid_type():
            return cls(uuid.UUID(val), col_type)
        if col_type.is_binary_type():
            assert isinstance(val, str)
            bytes_val = base64.b64decode(val.encode('utf-8'))
            return cls(bytes_val, col_type)
        if col_type.is_array_type():
            if TYPE_CHECKING:
                assert isinstance(col_type, ts.ArrayType)
            dtype = col_type.dtype  # possibly None
            array = np.array(val, dtype=dtype)
            return cls(array, col_type=col_type)
        # For all other types, val should already be in the right format
        return cls(val, col_type)
