from typing import Iterable, Union, Optional

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

    def get(self, e: Expr) -> Optional[sql.ColumnElement]:
        """Returns the sql.ColumnElement for the given Expr, or None if Expr.to_sql() returns None."""
        try:
            return self.cache[e.id]
        except KeyError:
            pass
        el = e.sql_expr(self)
        self.cache[e.id] = el
        return el

    def contains(self, items: Union[Expr, Iterable[Expr]]) -> bool:
        """Returns True if every item has a (non-None) sql.ColumnElement."""
        if isinstance(items, Expr):
            return self.get(items) is not None
        return all(self.get(e) is not None for e in items)
