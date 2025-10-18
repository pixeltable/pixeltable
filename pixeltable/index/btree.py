from typing import TYPE_CHECKING, Optional

import sqlalchemy as sql

# TODO: why does this import result in a circular import, but the one im embedding_index.py doesn't?
# import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import catalog, exprs
from pixeltable.env import Env
from pixeltable.func.udf import udf

from .base import IndexBase

if TYPE_CHECKING:
    from pixeltable import catalog, exprs


class BtreeIndex(IndexBase):
    """
    Interface to B-tree indices in Postgres.
    """

    MAX_STRING_LEN = 256

    @staticmethod
    @udf
    def str_filter(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        return s[: BtreeIndex.MAX_STRING_LEN]

    def __init__(self) -> None:
        pass

    def create_value_expr(self, c: 'catalog.Column') -> 'exprs.Expr':
        if not c.col_type.is_scalar_type() and not c.col_type.is_media_type():
            raise excs.Error(f'Index on column {c.name}: B-tree index requires scalar or media type, got {c.col_type}')
        value_expr: exprs.Expr
        if c.col_type.is_media_type():
            # an index on a media column is an index on the file url
            # no validation for media columns: we're only interested in the string value
            value_expr = exprs.ColumnRef(c, perform_validation=False)
        else:
            value_expr = (
                BtreeIndex.str_filter(exprs.ColumnRef(c)) if c.col_type.is_string_type() else exprs.ColumnRef(c)
            )
        return value_expr

    def records_value_errors(self) -> bool:
        return False

    def get_index_sa_type(self, col_type: ts.ColumnType) -> sql.types.TypeEngine:
        """Return the sqlalchemy type of the index value column"""
        return col_type.to_sa_type()

    def create_index(self, index_name: str, index_value_col: 'catalog.Column') -> None:
        """Create the index on the index value column"""
        idx = sql.Index(index_name, index_value_col.sa_col, postgresql_using='btree')
        conn = Env.get().conn
        idx.create(bind=conn)

    def drop_index(self, index_name: str, index_value_col: 'catalog.Column') -> None:
        """Drop the index on the index value column"""
        # TODO: implement
        raise NotImplementedError()

    @classmethod
    def display_name(cls) -> str:
        return 'btree'

    def as_dict(self) -> dict:
        return {}

    @classmethod
    def from_dict(cls, d: dict) -> 'BtreeIndex':
        return cls()
