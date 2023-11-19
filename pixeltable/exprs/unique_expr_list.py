from __future__ import annotations
from typing import Optional, List, Iterable, Type, Iterator

from .expr import Expr


class UniqueExprList:
    """
    A List[Expr] which ignores duplicates and which supports [] access by Expr.equals().
    We can't use set() because Expr doesn't have a __hash__() and Expr.__eq__() has been repurposed.

    TODO: now that we have Expr.id, replace with UniqueExprs, implemented as a Dict[Expr.id, Expr]
    """
    def __init__(self, elements: Optional[Iterable[Expr]] = None):
        self.exprs: List[Expr] = []
        if elements is not None:
            for e in elements:
                self.append(e)

    def append(self, expr: Expr) -> None:
        try:
            _ = next(e for e in self.exprs if e.equals(expr))
        except StopIteration:
            self.exprs.append(expr)

    def extend(self, elements: Iterable[Expr]) -> None:
        for e in elements:
            self.append(e)

    def __contains__(self, item: Expr) -> bool:
        assert isinstance(item, Expr)
        try:
            _ = next(e for e in self.exprs if e.equals(item))
            return True
        except StopIteration:
            return False

    def contains(self, cls: Type[Expr]) -> bool:
        try:
            _ = next(e for e in self.exprs if isinstance(e, cls))
            return True
        except StopIteration:
            return False

    def __len__(self) -> int:
        return len(self.exprs)

    def __iter__(self) -> Iterator[Expr]:
        return iter(self.exprs)

    def __getitem__(self, index: object) -> Optional[Expr]:
        assert isinstance(index, int) or isinstance(index, Expr)
        if isinstance(index, int):
            # return expr with matching slot_idx
            return [e for e in self.exprs if e.slot_idx == index][0]
        else:
            try:
                return next(e for e in self.exprs if e.equals(index))
            except StopIteration:
                return None
