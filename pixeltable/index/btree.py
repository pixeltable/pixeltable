from typing import Optional, TYPE_CHECKING

import sqlalchemy as sql

# TODO: why does this import result in a circular import, but the one im embedding_index.py doesn't?
# import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
from pixeltable import catalog, exprs
from pixeltable.func.udf import udf
from .base import IndexBase

if TYPE_CHECKING:
    import pixeltable

@udf
def _str_filter(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return s[:BtreeIndex.MAX_STRING_LEN]


class BtreeIndex(IndexBase):
    """
    Interface to B-tree indices in Postgres.
    """
    MAX_STRING_LEN = 256

    value_expr: 'pixeltable.exprs.Expr'

    @staticmethod
    @udf
    def str_filter(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        return s[:BtreeIndex.MAX_STRING_LEN]

    def __init__(self, c: 'catalog.Column'):
        if not c.col_type.is_scalar_type() and not c.col_type.is_media_type():
            raise excs.Error(f'Index on column {c.name}: B-tree index requires scalar or media type, got {c.col_type}')
        from pixeltable.exprs import ColumnRef, ColumnPropertyRef
        if c.col_type.is_media_type():
            # an index on a media column is an index on the file url
            # no validation for media columns: we're only interested in the string value
            self.value_expr = ColumnRef(c, perform_validation=False)
        else:
            self.value_expr = BtreeIndex.str_filter(ColumnRef(c)) if c.col_type.is_string_type() else ColumnRef(c)

    def index_value_expr(self) -> 'exprs.Expr':
        return self.value_expr

    def records_value_errors(self) -> bool:
        return False

    def index_sa_type(self) -> sql.types.TypeEngine:
        """Return the sqlalchemy type of the index value column"""
        return self.value_expr.col_type.to_sa_type()

    def create_index(self, index_name: str, index_value_col: 'catalog.Column', conn: sql.engine.Connection) -> None:
        """Create the index on the index value column"""
        idx = sql.Index(index_name, index_value_col.sa_col, postgresql_using='btree')
        idx.create(bind=conn)

    @classmethod
    def display_name(cls) -> str:
        return 'btree'

    def as_dict(self) -> dict:
        return {}

    @classmethod
    def from_dict(cls, c: 'catalog.Column', d: dict) -> 'BtreeIndex':
        return cls(c)

