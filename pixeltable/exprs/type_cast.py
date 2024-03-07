import json
from typing import Optional, Dict, List, Tuple, Any

import sqlalchemy as sql

import pixeltable.type_system as ts
from .expr import DataRow, Expr
from .row_builder import RowBuilder


class TypeCast(Expr):
    """
    An `Expr` that represents an explicit type conversion from an underlying `Expr` to
    a specified `ColumnType`.

    Currently, the only type conversions supported are:
    - `StringType` to `JsonType`
    - Any `T` to `StringType`
    - Any `T` to a supertype of `T`
    This may be extended in the future.
    """
    # TODO: Support other type conversions as needed.
    def __init__(self, underlying: Expr, new_type: ts.ColumnType):
        super().__init__(new_type)
        if self.col_type.is_supertype_of(underlying.col_type) \
                or self.col_type.is_string_type() \
                or self.col_type.is_json_type() and underlying.col_type.is_string_type():
            # It's a valid type conversion
            self.components: List[Expr] = [underlying]
            self.id: Optional[int] = self._create_id()
        else:
            raise RuntimeError(f'Expression of type `{self._underlying.col_type}` cannot be cast to `{self.col_type}`')

    @property
    def _underlying(self):
        return self.components[0]

    def _equals(self, other: Expr) -> bool:
        # `TypeCast` has no properties beyond those captured by `Expr`.
        return True

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('new_type', self.col_type)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        """
        `sql_expr` is unimplemented for now, in order to sidestep potentially thorny
        questions about consistency of doing type conversions in both Python and Postgres.
        """
        return None
        # underlying_sql_expr = self._underlying.sql_expr()
        # if underlying_sql_expr is None:
        #     return None
        # else:
        #     return underlying_sql_expr.cast(self.col_type.to_sa_type)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        original_val = data_row[self._underlying.slot_idx]
        if self.col_type.is_supertype_of(self._underlying.col_type):
            # No conversion necessary
            cast_val = original_val
        elif self.col_type.is_string_type() and self._underlying.col_type.is_json_type():
            # For converting JSON -> string, we need to use json.dumps, not call str(dict)
            cast_val = json.dumps(original_val)
        elif self.col_type.is_string_type():
            cast_val = str(original_val)
        elif self.col_type.is_json_type() and self._underlying.col_type.is_string_type():
            cast_val = json.loads(original_val)
        else:
            assert False    # This should have been caught in __init__
        data_row[self.slot_idx] = cast_val

    def _as_dict(self) -> Dict:
        return {'new_type': self.col_type.as_dict(), **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'new_type' in d
        assert len(components) == 1
        return cls(components[0], ts.ColumnType.from_dict(d['new_type']))

    def __str__(self) -> str:
        return f'{self._underlying}.astype({self.col_type})'
