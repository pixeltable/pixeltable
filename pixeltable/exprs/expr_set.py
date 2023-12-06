from __future__ import annotations
from typing import Optional, Dict, Iterable, Iterator

from .expr import Expr


class ExprSet:
    """A set that also supports indexed lookup (by slot_idx and Expr.id)"""
    def __init__(self, elements: Optional[Iterable[Expr]] = None):
        self.exprs: Dict[int, Expr] = {}  # Expr.id -> Expr
        if elements is not None:
            for e in elements:
                self.append(e)

    def append(self, expr: Expr) -> None:
        if expr.id in self.exprs:
            return
        self.exprs[expr.id] = expr

    def extend(self, elements: Iterable[Expr]) -> None:
        for e in elements:
            self.append(e)

    def __contains__(self, item: Expr) -> bool:
        return item.id in self.exprs

    def __len__(self) -> int:
        return len(self.exprs)

    def __iter__(self) -> Iterator[Expr]:
        return iter(self.exprs.values())

    def __getitem__(self, index: object) -> Optional[Expr]:
        assert isinstance(index, int) or isinstance(index, Expr)
        if isinstance(index, int):
            # return expr with matching slot_idx
            return list(self.exprs.values())[index]
        else:
            return self.exprs.get(index.id)
