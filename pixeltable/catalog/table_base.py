from __future__ import annotations

import json
import logging
from typing import Union, Any
from uuid import UUID

import pandas as pd
import sqlalchemy as sql

from .schema_object import SchemaObject
from .table_version import TableVersion
from pixeltable import exceptions as exc
from pixeltable.env import Env
from pixeltable.metadata import schema

_ID_RE = r'[a-zA-Z]\w*'
_PATH_RE = f'{_ID_RE}(\\.{_ID_RE})*'


_logger = logging.getLogger('pixeltable')

class TableBase(SchemaObject):
    """Base class for all schema objects that can be queried."""
    def __init__(self, id: UUID, dir_id: UUID, name: str, tbl_version: TableVersion):
        super().__init__(id, name, dir_id)
        self.is_dropped = False
        self.tbl_version = tbl_version

    def _check_is_dropped(self) -> None:
        if self.is_dropped:
            raise exc.Error(f'{self.display_name()} {self.name} has been dropped')

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        super().move(new_name, new_dir_id)
        with Env.get().engine.begin() as conn:
            stmt = sql.text((
                f"UPDATE {schema.Table.__table__} "
                f"SET {schema.Table.dir_id.name} = :new_dir_id, "
                f"    {schema.Table.md.name}['name'] = :new_name "
                f"WHERE {schema.Table.id.name} = :id"))
            conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self.id})

    def __getattr__(self, col_name: str) -> 'pixeltable.exprs.ColumnRef':
        """Return a ColumnRef for the given column name.
        """
        return getattr(self.tbl_version, col_name)

    def __getitem__(self, index: object) -> Union['pixeltable.exprs.ColumnRef', 'pixeltable.dataframe.DataFrame']:
        """Return a ColumnRef for the given column name, or a DataFrame for the given slice.
        """
        return self.tbl_version.__getitem__(index)

    def df(self) -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version)

    def select(self, *items: Any, **named_items : Any) -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version).select(*items, **named_items)

    def where(self, pred: 'exprs.Predicate') -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version).where(pred)

    def order_by(self, *items: 'exprs.Expr', asc: bool = True) -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version).order_by(*items, asc=asc)

    def show(
            self, *args, **kwargs
    ) -> 'pixeltable.dataframe.DataFrameResultSet':  # type: ignore[name-defined, no-untyped-def]
        """Return rows from this table.
        """
        return self.df().show(*args, **kwargs)

    def head(
            self, *args, **kwargs
    ) -> 'pixeltable.dataframe.DataFrameResultSet':  # type: ignore[name-defined, no-untyped-def]
        """Return rows from this table.
        """
        return self.df().head(*args, **kwargs)

    def count(self) -> int:
        """Return the number of rows in this table.
        """
        return self.df().count()

    def _description(self) -> pd.DataFrame:
        return pd.DataFrame({
            'Column Name': [c.name for c in self.cols],
            'Type': [str(c.col_type) for c in self.cols],
            'Computed With':
                [c.value_expr.display_str(inline=False) if c.value_expr is not None else '' for c in self.cols],
        })

    def _description_html(self) -> pd.DataFrame:
        pd_df = self._description()
        # white-space: pre-wrap: print \n as newline
        # th: center-align headings
        return pd_df.style.set_properties(**{'white-space': 'pre-wrap', 'text-align': 'left'}) \
            .set_table_styles([dict(selector='th', props=[('text-align', 'center')])]) \
            .hide(axis='index')

    def describe(self) -> None:
        try:
            __IPYTHON__
            from IPython.display import display
            display(self._description_html())
        except NameError:
            print(self.__repr__())

    def __repr__(self) -> str:
        return self._description().to_string(index=False)

    def _repr_html_(self) -> str:
        return self._description_html()._repr_html_()

    def drop(self) -> None:
        self._check_is_dropped()
        self.is_dropped = True
        self.tbl_version.drop()

