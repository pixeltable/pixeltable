from typing import Tuple, List
import pandas as pd
import sqlalchemy as sql

from pixeltable import catalog, store
from pixeltable import type_system as pt_types

__all__ = [
    'DataFrame'
]


class DataFrameResultSet:
    def __init__(self, rows: Tuple, col_names: List[str], col_types: List[pt_types.ColumnType]):
        self.rows = rows
        self.col_names = col_names
        self.col_types = col_types

    def _repr_html_(self) -> str:
        return self._create_df().to_html()

    def __str__(self) -> str:
        return self._create_df().to_string()

    def _create_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.rows, columns=self.col_names)


class DataFrame:
    def __init__(self, tbl: catalog.Table):
        self.tbl = tbl
        self.selected_cols = tbl.cols.copy()

    def show(self, num_rows: int = 20) -> DataFrameResultSet:
        with store.engine.connect() as conn:
            stmt = self._create_select_stmt()
            i = 0
            rows = []
            for row in conn.execute(stmt):
                rows.append(row)
                i += 1
                if i == num_rows:
                    break
        return DataFrameResultSet(
            rows, [col.name for col in self.selected_cols], [col.col_type for col in self.selected_cols])

    def _create_select_stmt(self) -> sql.sql.expression.Select:
        selected_sa_cols = [col.sa_col for col in self.selected_cols]
        stmt = sql.select(*selected_sa_cols) \
            .where(self.tbl.v_min_col <= self.tbl.version) \
            .where(self.tbl.v_max_col > self.tbl.version)
        return stmt
