from typing import TYPE_CHECKING

import sqlalchemy as sql

# TODO: why does this import result in a circular import, but the one im embedding_index.py doesn't?
# import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
import pixeltable.type_system as ts
from pixeltable.func.udf import udf

from .base import IndexBase

if TYPE_CHECKING:
    import pixeltable.catalog as catalog


class BtreeIndex(IndexBase):
    """
    Interface to B-tree indices in Postgres.

    Two representations, selected by uses_value_col:
    - with a value column, Postgres indexes a dedicated index value column, which for a string column holds the value
      truncated to MAX_STRING_LEN; expired rows are excluded from the index by nulling that column
    - without one, Postgres indexes the stored column directly, with no truncation
    """

    MAX_STRING_LEN = 256

    @staticmethod
    @udf
    def str_filter(s: str | None) -> str | None:
        if s is None:
            return None
        return s[: BtreeIndex.MAX_STRING_LEN]

    def __init__(self, uses_value_col: bool) -> None:
        self._uses_value_col = uses_value_col

    @property
    def uses_value_col(self) -> bool:
        return self._uses_value_col

    @classmethod
    def can_index(cls, c: 'catalog.Column') -> bool:
        """True if c is eligible for a B-tree index, based on its type and other properties."""
        try:
            cls.validate_column(c)
        except excs.RequestError:
            return False
        return True

    @classmethod
    def validate_column(cls, c: 'catalog.Column') -> None:
        """Raises if c isn't eligible for a B-tree index, based on its type and other properties."""
        if not c.stored:
            # if the column is intentionally not stored, we want to avoid the overhead of an index
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot create a B-tree index on unstored column {c.name!r}.'
            )
        # bools are excluded: a B-tree on a two-valued column isn't useful
        if not ((c.col_type.is_scalar_type() and not c.col_type.is_bool_type()) or c.col_type.is_media_type()):
            raise excs.RequestError(
                excs.ErrorCode.TYPE_MISMATCH,
                f'Index on column {c.name}: B-tree index requires a non-boolean scalar type or a media type, '
                f'got {c.col_type}',
            )
        if c.col_type.is_media_type():
            if c.is_iterator_col:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot create a B-tree index on column {c.name!r}, which is produced by an iterator.',
                )
            if c.is_computed:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot create a B-tree index on computed media column {c.name!r}.',
                )

    def create_value_expr(self, c: 'catalog.Column') -> 'exprs.Expr':
        assert self.uses_value_col
        self.validate_column(c)
        col_md = c.column_version_md()
        value_expr: exprs.Expr
        col_ref = exprs.ColumnRef(col_md)
        if c.col_type.is_media_type():
            # an index on a media column is an index on the file url
            # no validation for media columns: we're only interested in the string value
            value_expr = exprs.ColumnRef(col_md, perform_validation=False)
        else:
            value_expr = BtreeIndex.str_filter(col_ref) if c.col_type.is_string_type() else col_ref
        return value_expr

    def records_value_errors(self) -> bool:
        return False

    def get_index_sa_type(self, val_col_type: ts.ColumnType) -> sql.types.TypeEngine:
        """Return the sqlalchemy type of the index value column"""
        assert self.uses_value_col
        return val_col_type.to_sa_type()

    def sa_create_stmt(self, store_index_name: str, sa_col: sql.Column) -> sql.Compiled:
        """Return a sqlalchemy statement for creating the index on sa_col."""
        from sqlalchemy.dialects import postgresql

        sa_idx = sql.Index(store_index_name, sa_col, postgresql_using='btree')
        return sql.schema.CreateIndex(sa_idx, if_not_exists=True).compile(dialect=postgresql.dialect())

    @classmethod
    def display_name(cls) -> str:
        return 'btree'

    def as_dict(self) -> dict:
        return {'uses_value_col': self._uses_value_col}

    @classmethod
    def from_dict(cls, d: dict) -> 'BtreeIndex':
        return cls(uses_value_col=d.get('uses_value_col', True))
