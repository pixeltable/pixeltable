from __future__ import annotations

from typing import Generic, Iterable, Iterator, Optional, TypeVar

from .expr import Expr

T = TypeVar('T', bound='Expr')


class ExprSet(Generic[T]):
    """
    A set that also supports indexed lookup (by slot_idx and Expr.id). Exprs are uniquely identified by Expr.id.
    """

    exprs: dict[int, T]  # key: Expr.id
    exprs_by_idx: dict[int, T]  # key: slot_idx

    def __init__(self, elements: Optional[Iterable[T]] = None):
        self.exprs = {}
        self.exprs_by_idx = {}
        if elements is not None:
            for e in elements:
                self.add(e)

    def add(self, expr: T) -> None:
        if expr.id in self.exprs:
            return
        self.exprs[expr.id] = expr
        if expr.slot_idx is None:
            return
        self.exprs_by_idx[expr.slot_idx] = expr

    def update(self, *others: Iterable[T]) -> None:
        for other in others:
            for e in other:
                self.add(e)

    def __contains__(self, item: T) -> bool:
        return item.id in self.exprs

    def __len__(self) -> int:
        return len(self.exprs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.exprs.values())

    def __getitem__(self, index: object) -> Optional[T]:
        """Indexed lookup by slot_idx or Expr.id."""
        assert isinstance(index, (int, Expr))
        if isinstance(index, int):
            # return expr with matching slot_idx
            return self.exprs_by_idx.get(index)
        else:
            return self.exprs.get(index.id)

    def issuperset(self, other: ExprSet[T]) -> bool:
        return self.exprs.keys() >= other.exprs.keys()

    def __ge__(self, other: ExprSet[T]) -> bool:
        return self.issuperset(other)

    def __le__(self, other: ExprSet[T]) -> bool:
        return other.issuperset(self)

    def union(self, *others: Iterable[T]) -> ExprSet[T]:
        result = ExprSet(self.exprs.values())
        result.update(*others)
        return result

    def __or__(self, other: ExprSet[T]) -> ExprSet[T]:
        return self.union(other)

    def difference(self, *others: Iterable[T]) -> ExprSet[T]:
        id_diff = set(self.exprs.keys()).difference(e.id for other_set in others for e in other_set)
        return ExprSet(self.exprs[id] for id in id_diff)

    def __sub__(self, other: ExprSet[T]) -> ExprSet[T]:
        return self.difference(other)

    def __add__(self, other: ExprSet) -> ExprSet:
        exprs = self.exprs.copy()
        exprs.update(other.exprs)
        return ExprSet(exprs.values())
