from typing import Any

from pixeltable import type_system as ts

from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder


class UnknownExpr(Expr):
    """
    Represents an expression for which the specific expression type cannot yet be determined, because it contains
    a subexpression of ambiguous type.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(ts.InvalidType(nullable=False))

    def __call__(self, *args: Any, **kwargs: Any) -> 'UnknownCallExpr':
        return UnknownCallExpr(self, args, kwargs)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise AssertionError('It should never be possible to observe an UnknownExpr in an execution context.')


class UnknownOpExpr(UnknownExpr):
    def __init__(self, op: str, operand1: Expr, operand2: Any):
        super().__init__()
        self.op = op
        self.operand1 = operand1
        self.operand2 = operand2

    def substitute(self, spec: dict[Expr, Expr]) -> Expr:
        operand1_sub = self.operand1.substitute(spec)
        operand2_sub = self.operand2.substitute(spec) if isinstance(self.operand2, Expr) else self.operand2
        return getattr(operand1_sub, self.op)(operand2_sub)


class UnknownItemExpr(UnknownExpr):
    def __init__(self, base: Expr, index: object):
        super().__init__()
        self.base = base
        self.index = index

    def substitute(self, spec: dict[Expr, Expr]) -> Expr:
        base_sub = self.base.substitute(spec)
        return base_sub[self.index]


class UnknownAttrExpr(UnknownExpr):
    def __init__(self, base: Expr, attr_name: str):
        super().__init__()
        self.base = base
        self.attr_name = attr_name

    def substitute(self, spec: dict[Expr, Expr]) -> Expr:
        base_sub = self.base.substitute(spec)
        return getattr(base_sub, self.attr_name)


class UnknownCallExpr(UnknownExpr):
    def __init__(self, base: Expr, args: tuple[Any, ...], kwargs: dict[str, Any]):
        super().__init__()
        self.base = base
        self.args = args
        self.kwargs = kwargs

    def substitute(self, spec: dict[Expr, Expr]) -> Expr:
        base_sub = self.base.substitute(spec)
        args_sub = tuple(arg.substitute(spec) if isinstance(arg, Expr) else arg for arg in self.args)
        kwargs_sub = {k: v.substitute(spec) if isinstance(v, Expr) else v for k, v in self.kwargs.items()}
        return base_sub(*args_sub, **kwargs_sub)
