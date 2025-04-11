from typing import Iterable, Optional

import sqlalchemy as sql

from .expr import Expr
from .expr_dict import ExprDict


class SqlElementCache:
    """Cache of sql.ColumnElements for exprs"""

    cache: dict[int, Optional[sql.ColumnElement]]  # key: Expr.id

    def __init__(self, elements: Optional[ExprDict[sql.ColumnElement]] = None):
        self.cache = {}
        if elements is not None:
            for e, el in elements.items():
                self.cache[e.id] = el

    def extend(self, elements: ExprDict[sql.ColumnElement]) -> None:
        for e, el in elements.items():
            self.cache[e.id] = el

    def get(self, e: Expr) -> Optional[sql.ColumnElement]:
        """Returns the sql.ColumnElement for the given Expr, or None if Expr.to_sql() returns None."""
        try:
            return self.cache[e.id]
        except KeyError:
            pass
        el = e.sql_expr(self)
        self.cache[e.id] = el
        return el

    def contains(self, item: Expr) -> bool:
        """Returns True if the cache contains a (non-None) value for the given Expr."""
        return self.get(item) is not None

    def contains_all(self, items: Iterable[Expr]) -> bool:
        """Returns True if the cache contains a (non-None) value for every item in the collection of Exprs."""
        return all(self.get(e) is not None for e in items)
