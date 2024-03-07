from __future__ import annotations
from typing import Optional, List, Tuple, Callable

from .expr import Expr
from .globals import LogicalOperator
import pixeltable
import pixeltable.type_system as ts


class Predicate(Expr):
    def __init__(self) -> None:
        super().__init__(ts.BoolType())

    def split_conjuncts(
            self, condition: Callable[[Predicate], bool]) -> Tuple[List[Predicate], Optional[Predicate]]:
        """
        Returns clauses of a conjunction that meet condition in the first element.
        The second element contains remaining clauses, rolled into a conjunction.
        """
        if condition(self):
            return [self], None
        else:
            return [], self

    def __and__(self, other: object) -> 'pixeltable.exprs.CompoundPredicate':
        if not isinstance(other, Expr):
            raise TypeError(f'Other needs to be an expression: {type(other)}')
        if not other.col_type.is_bool_type():
            raise TypeError(f'Other needs to be an expression that returns a boolean: {other.col_type}')
        from .compound_predicate import CompoundPredicate
        return CompoundPredicate(LogicalOperator.AND, [self, other])

    def __or__(self, other: object) -> 'pixeltable.exprs.CompoundPredicate':
        if not isinstance(other, Expr):
            raise TypeError(f'Other needs to be an expression: {type(other)}')
        if not other.col_type.is_bool_type():
            raise TypeError(f'Other needs to be an expression that returns a boolean: {other.col_type}')
        from .compound_predicate import CompoundPredicate
        return CompoundPredicate(LogicalOperator.OR, [self, other])

    def __invert__(self) -> 'pixeltable.exprs.CompoundPredicate':
        from .compound_predicate import CompoundPredicate
        return CompoundPredicate(LogicalOperator.NOT, [self])

