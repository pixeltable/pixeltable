from typing import Iterable, Optional

import sqlalchemy as sql

from .expr import Expr


class SqlElementCache:
    """Cache of sql.ColumnElements for exprs"""

    cache: dict[int, sql.ColumnElement]  # key: Expr.id

    def __init__(self):
        self.cache = {}

    def __getitem__(self, e: Expr) -> Optional[sql.ColumnElement]:
        try:
            return self.cache[e.id]
        except KeyError:
            pass
        el = e.sql_expr(self)
        self.cache[e.id] = el
        return el

    def contains(self, expr_set: Iterable[Expr]) -> bool:
        return all(self[e] is not None for e in expr_set)
