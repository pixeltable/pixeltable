from typing import Any, Optional

import sqlalchemy as sql

import pixeltable.type_system as ts
from pixeltable.exprs import Expr, FunctionCall
from pixeltable.func import FunctionRegistry

from .data_row import DataRow
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class MethodRef(Expr):
    """
    A method reference. This represents a `Function` instance with its
    first parameter bound to a base expression.

    When a `MethodRef` is called, it returns a `FunctionCall` with the base expression as the first argument.
    The effective arity of a `MethodRef` is one less than the arity of the underlying `Function`.
    """

    # TODO: Should this even be an `Expr`? It can't actually be evaluated directly (it has to be first
    #   converted to a `FunctionCall` by binding any remaining parameters).

    def __init__(self, base_expr: Expr, method_name: str) -> None:
        super().__init__(ts.InvalidType())  # The `MethodRef` is untyped until it is called.
        self.base_expr = base_expr
        self.method_name = method_name
        self.fn = FunctionRegistry.get().lookup_type_method(base_expr.col_type.type_enum, method_name)
        if self.fn is None:
            # This has to be an `AttributeError`, or tab-completion won't work properly in ipython.
            raise AttributeError(f'Unknown method (of type {base_expr.col_type}): {method_name}')
        self.components = [base_expr]
        self.id = self._create_id()

    def _as_dict(self) -> dict:
        return {'method_name': self.method_name, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> 'MethodRef':
        assert 'method_name' in d
        assert len(components) == 1
        return cls(components[0], d['method_name'])

    def __call__(self, *args: Any, **kwargs: Any) -> FunctionCall:
        result = self.fn(*[self.base_expr, *args], **kwargs)
        assert isinstance(result, FunctionCall)
        result.is_method_call = True
        return result

    def _equals(self, other: 'MethodRef') -> bool:
        return self.base_expr.id == other.base_expr.id and self.method_name == other.method_name

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('method_name', self.method_name)]

    def sql_expr(self, _: SqlElementCache) -> Optional[sql.ColumnElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise AssertionError('MethodRef cannot be evaluated directly')

    def __repr__(self) -> str:
        return f'{self.base_expr}.{self.method_name}'
