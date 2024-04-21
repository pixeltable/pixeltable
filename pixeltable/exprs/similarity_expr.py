from typing import Optional, List

import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .literal import Literal
from .row_builder import RowBuilder


class SimilarityExpr(Expr):

    def __init__(self, left: ColumnRef, right: Expr):
        super().__init__(ts.FloatType())
        self.components = [left, right]
        self.id = self._create_id()
        # determine index to use
        idx_info = left.col.get_idx_info()
        if len(idx_info) == 0:
            raise excs.Error(f'No index found for column {left.col}')
        if len(idx_info) > 1:
            raise excs.Error(
                f'Column {left.col} has multiple indices; use the index name to disambiguate, '
                f'e.g., "{left.col}.<index-name> %% ..."')
        self.idx_info = next(iter(idx_info.values()))

    def __str__(self) -> str:
        return f'{self.components[0]} %% {self.components[1]}'

    def _equals(self, other: Expr) -> bool:
        return self.components == other.components

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        assert isinstance(self.components[1], Literal)
        item = self.components[1].val
        return self.idx_info.idx.search_clause(self.idx_info.val_col, item)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        pass

    @classmethod
    def _from_dict(cls, d: dict, components: List[Expr]) -> Expr:
        assert len(components) == 2
        return cls(components[0], components[1])
