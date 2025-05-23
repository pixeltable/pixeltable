from typing import Any, Optional

import sqlalchemy as sql

from pixeltable import type_system as ts
from pixeltable.exprs.expr import DataRow, Expr
from pixeltable.exprs.literal import Literal
from pixeltable.exprs.row_builder import RowBuilder
from pixeltable.exprs.sql_element_cache import SqlElementCache
from pixeltable.func.udf import udf


class SampleKey(Expr):
    """
    An `Expr` that computes a custom expression from the RowidRef's required to uniquely access a table
    """

    def __init__(self, seed_expr: Expr, rowids: list[Expr]):
        super().__init__(ts.StringType(nullable=True))
        self.components = [seed_expr, *rowids]
        self.id: Optional[int] = self._create_id()

    @property
    def _seed(self) -> Optional[int]:
        assert isinstance(self._seed_expr.val, (int, type(None)))
        return self._seed_expr.val

    @property
    def _seed_expr(self) -> Expr:
        return self.components[0]

    def _equals(self, other: 'SampleKey') -> bool:
        return True

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise NotImplementedError

    def as_literal(self) -> Optional[Literal]:
        return None

    def _as_dict(self) -> dict:
        return {**super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> 'SampleKey':
        assert len(components) >= 2
        return cls(components[0], components[1:])

    def __repr__(self) -> str:
        return ''

    def sql_expr(self, sql_elements: SqlElementCache) -> Optional[sql.ColumnElement]:
        """Construct expression which is the ordering key for rows to be sampled
        General SQL form is:
        - MD5('<seed::text>' [ + '___' + <rowid_col_val>::text]+
        """
        from pixeltable.plan import SampleClause

        rowid_sql_expr = [e.sql_expr(sql_elements) for e in self.components[1:]]
        return SampleClause.key_sql_expr(self._seed, rowid_sql_expr)


@udf
def sample_key(seed: int, rowid_sql_exprs: ts.Json) -> str:
    """
    Create a sample key from the given seed and rowids.

    Args:
        seed: The seed value.
        rowids: The rowids to include in the sample key.

    Returns:
        A SampleKey object.
    """
    assert isinstance(seed, int)
    raise NotImplementedError


@sample_key.to_sql
def _(seed: int, rowid_sql_exprs: sql.ColumnCollection[Any, Any]) -> sql.ColumnElement:
    from pixeltable.plan import SampleClause

    return SampleClause.key_sql_expr(seed, list(rowid_sql_exprs))
