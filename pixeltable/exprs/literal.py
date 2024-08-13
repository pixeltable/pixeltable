from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

import sqlalchemy as sql

import pixeltable.type_system as ts
from pixeltable.env import Env

from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder


class Literal(Expr):
    def __init__(self, val: Any, col_type: Optional[ts.ColumnType] = None):
        if col_type is not None:
            val = col_type.create_literal(val)
        else:
            # try to determine a type for val
            col_type = ts.ColumnType.infer_literal_type(val)
            if col_type is None:
                raise TypeError(f'Not a valid literal: {val}')
        super().__init__(col_type)
        self.val = val
        self.id = self._create_id()

    def default_column_name(self) -> Optional[str]:
        return 'Literal'

    def __str__(self) -> str:
        if self.col_type.is_string_type() or self.col_type.is_timestamp_type():
            return f"'{self.val}'"
        return str(self.val)

    def _equals(self, other: Literal) -> bool:
        return self.val == other.val

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('val', self.val)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        # we need to return something here so that we can generate a Where clause for predicates
        # that involve literals (like Where c > 0)
        return sql.sql.expression.literal(self.val)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        # this will be called, even though sql_expr() does not return None
        data_row[self.slot_idx] = self.val

    def _as_dict(self) -> Dict:
        # For some types, we need to explictly record their type, because JSON does not know
        # how to interpret them unambiguously
        if self.col_type.is_timestamp_type():
            assert isinstance(self.val, datetime.datetime)
            # Convert to ISO format in UTC (in keeping with the principle: all timestamps are
            # stored as UTC in the database)
            encoded_val = self.val.astimezone(datetime.timezone.utc).isoformat()
            return {'val': encoded_val, 'val_t': self.col_type._type.name, **super()._as_dict()}
        else:
            return {'val': self.val, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        print(f'Found a literal: {d}')
        assert 'val' in d
        if 'val_t' in d:
            val_t = d['val_t']
            # Currently the only special-cased literal type is TIMESTAMP
            assert val_t == ts.ColumnType.Type.TIMESTAMP.name
            dt = datetime.datetime.fromisoformat(d['val'])
            assert dt.tzinfo == datetime.timezone.utc  # Must be UTC in the database
            # Convert from UTC to the default time zone (which may be none, in which case, use system time zone)
            return cls(dt.astimezone(Env.get().default_time_zone))
        else:
            return cls(d['val'])
