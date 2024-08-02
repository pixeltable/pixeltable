from typing import Any, Optional

import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.exprs import Expr, FunctionCall
from pixeltable.func import FunctionRegistry, CallableFunction
from .data_row import DataRow
from .row_builder import RowBuilder


class MemberRef(Expr):
    """
    A method reference. This represents access to an unevaluated method of a Pixeltable expression, such as
    `t.img_col.rotate` or `t.str_col.contains`. Equivalently, it represents a `Function` instance with its
    first parameter bound to a base expression.
    """

    def __init__(self, base_expr: Expr, member_name: str):
        super().__init__(ts.InvalidType())  # type is not defined until it's instantiated
        self.base_expr = base_expr
        self.member_name = member_name
        self.fn = FunctionRegistry.get().lookup_type_method(base_expr.col_type.type_enum, member_name)
        if self.fn is None:
            raise excs.Error(f'Unknown member (of type {base_expr.col_type}): {member_name}')
        self.components = [base_expr]
        self.id = self._create_id()

    def _as_dict(self) -> dict:
        return {'member_name': self.member_name, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> Expr:
        assert 'member_name' in d
        assert len(components) == 1
        return cls(d['member_name'], components[0])

    def __call__(self, *args, **kwargs) -> FunctionCall:
        result = self.fn(*[self.base_expr, *args], **kwargs)
        assert isinstance(result, FunctionCall)
        result.is_method_call = True
        return result

    def _equals(self, other: 'MemberRef') -> bool:
        return self.member_name == other.member_name

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return super()._id_attrs() + [('member_name', self.member_name)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        base_val = data_row[self.base_expr.slot_idx]
        try:
            data_row[self.slot_idx] = getattr(base_val, self.member_name)
        except AttributeError:
            data_row[self.slot_idx] = None

    def __str__(self) -> str:
        return f'{self.base_expr}.{self.member_name}'
