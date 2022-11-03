import base64
from io import BytesIO
from typing import Tuple, List, Any
import pandas as pd
import sqlalchemy as sql
from PIL import Image

from pixeltable import catalog, store
from pixeltable import type_system as pt_types

__all__ = [
    'DataFrame'
]


def _format_img(path: object) -> str:
    assert type(path) == str
    try:
        img = Image.open(path)
        img.thumbnail((128, 128))
        with BytesIO() as buffer:
            img.save(buffer, 'jpeg')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f'<img src="data:image/jpeg;base64,{img_base64}">'
    except:
        return f'Error reading {path}'


class DataFrameResultSet:
    def __init__(self, rows: List[Tuple], col_names: List[str], col_types: List[pt_types.ColumnType]):
        self.rows = rows
        self.col_names = col_names
        self.col_types = col_types

    def _repr_html_(self) -> str:
        img_col_idxs = [i for i, col_type in enumerate(self.col_types) if col_type == pt_types.ColumnType.IMAGE]
        formatters = {self.col_names[i]: _format_img for i in img_col_idxs}
        # escape=False: make sure <img> tags stay intact
        # TODO: why does mypy complain about formatters having an incorrect type?
        return self._create_df().to_html(formatters=formatters, escape=False)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return self._create_df().to_string()

    def _create_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.rows, columns=self.col_names)


class DataFrame:
    def __init__(self, tbl: catalog.Table):
        self.tbl = tbl
        self.selected_cols = tbl.cols.copy()

    def show(self, n: int = 20) -> DataFrameResultSet:
        with store.engine.connect() as conn:
            stmt = self._create_select_stmt()
            i = 0
            rows: List[Tuple] = []
            for row in conn.execute(stmt):
                # TODO: is there a cleaner way to get the data?
                rows.append(row._data)
                i += 1
                if n > 0 and i == n:
                    break
        return DataFrameResultSet(
            rows, [col.name for col in self.selected_cols], [col.col_type for col in self.selected_cols])

    def count(self) -> int:
        stmt = sql.select(sql.func.count('*')).select_from(self.tbl.sa_tbl) \
            .where(self.tbl.v_min_col <= self.tbl.version) \
            .where(self.tbl.v_max_col > self.tbl.version)
        with store.engine.connect() as conn:
            return conn.execute(stmt).scalar_one()

    def _create_select_stmt(self) -> sql.sql.expression.Select:
        selected_sa_cols = [col.sa_col for col in self.selected_cols]
        stmt = sql.select(*selected_sa_cols) \
            .where(self.tbl.v_min_col <= self.tbl.version) \
            .where(self.tbl.v_max_col > self.tbl.version)
        return stmt
