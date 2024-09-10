from __future__ import annotations

from typing import Optional, Iterable, Iterator

from .expr import Expr


class ExprSet:
    """
    A set that also supports indexed lookup (by slot_idx and Expr.id). Exprs are uniquely identified by Expr.id.
    """
    exprs: dict[int, Expr]  # key: Expr.id
    exprs_by_idx: dict[int, Expr]  # key: slot_idx

    def __init__(self, elements: Optional[Iterable[Expr]] = None):
        self.exprs = {}
        self.exprs_by_idx = {}
        if elements is not None:
            for e in elements:
                self.add(e)

    def add(self, expr: Expr) -> None:
        if expr.id in self.exprs:
            return
        self.exprs[expr.id] = expr
        if expr.slot_idx is None:
            return
        self.exprs_by_idx[expr.slot_idx] = expr

    def update(self, *others: Iterable[Expr]) -> None:
        for other in others:
            for e in other:
                self.add(e)

    def __contains__(self, item: Expr) -> bool:
        return item.id in self.exprs

    def __len__(self) -> int:
        return len(self.exprs)

    def __iter__(self) -> Iterator[Expr]:
        return iter(self.exprs.values())

    def __getitem__(self, index: object) -> Optional[Expr]:
        """Indexed lookup by slot_idx or Expr.id."""
        if not isinstance(index, int) and not isinstance(index, Expr):
            pass
        assert isinstance(index, int) or isinstance(index, Expr)
        if isinstance(index, int):
            # return expr with matching slot_idx
            return self.exprs_by_idx.get(index)
        else:
            return self.exprs.get(index.id)

    def issuperset(self, other: ExprSet) -> bool:
        return self.exprs.keys() >= other.exprs.keys()

    def __ge__(self, other: ExprSet) -> bool:
        return self.issuperset(other)

    def __le__(self, other: ExprSet) -> bool:
        return other.issuperset(self)
