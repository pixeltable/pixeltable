from typing import TYPE_CHECKING
from uuid import UUID

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
    """

    MAX_STRING_LEN = 256

    @staticmethod
    @udf
    def str_filter(s: str | None) -> str | None:
        if s is None:
            return None
        return s[: BtreeIndex.MAX_STRING_LEN]

    def __init__(self) -> None:
        pass

    @classmethod
    def validation_error(cls, c: 'catalog.Column', tbl_id: UUID) -> excs.RequestError | None:
        """Returns None if the table with id tbl_id can have a B-tree index on c, or an error explaining why not."""
        if c.tbl_handle.id != tbl_id:
            # PXT-1260 Allow views to create a b-tree index on a base column
            return excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Cannot create a B-tree index on column {c.name!r}: it belongs to a base table. '
                'Add the index to the base table instead.',
            )
        return cls._column_error(c)

    @classmethod
    def can_index(cls, c: 'catalog.Column', tbl_id: UUID) -> bool:
        """True if the table with id tbl_id can have a B-tree index on c."""
        return cls.validation_error(c, tbl_id) is None

    @classmethod
    def _column_error(cls, c: 'catalog.Column') -> excs.RequestError | None:
        """Returns None if c can have a B-tree index, based on its type and other properties, or an error explaining
        why not.
        """
        if not c.stored:
            # if the column is intentionally not stored, we want to avoid the overhead of an index
            return excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot create a B-tree index on unstored column {c.name!r}.'
            )
        # bools are excluded: a B-tree on a two-valued column isn't useful
        if not ((c.col_type.is_scalar_type() and not c.col_type.is_bool_type()) or c.col_type.is_media_type()):
            return excs.RequestError(
                excs.ErrorCode.TYPE_MISMATCH,
                f'Index on column {c.name}: B-tree index requires a non-boolean scalar type or a media type, '
                f'got {c.col_type}',
            )
        if c.col_type.is_media_type():
            if c.is_iterator_col:
                return excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot create a B-tree index on column {c.name!r}, which is produced by an iterator.',
                )
            if c.is_computed:
                return excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot create a B-tree index on computed media column {c.name!r}.',
                )
        return None

    def create_value_expr(self, c: 'catalog.Column') -> 'exprs.Expr':
        err = self._column_error(c)
        if err is not None:
            raise err
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
        return val_col_type.to_sa_type()

    def sa_create_stmt(self, store_index_name: str, sa_value_col: sql.Column) -> sql.Compiled:
        """Return a sqlalchemy statement for creating the index"""
        from sqlalchemy.dialects import postgresql

        sa_idx = sql.Index(store_index_name, sa_value_col, postgresql_using='btree')
        return sql.schema.CreateIndex(sa_idx, if_not_exists=True).compile(dialect=postgresql.dialect())

    @classmethod
    def display_name(cls) -> str:
        return 'btree'

    def as_dict(self) -> dict:
        return {}

    @classmethod
    def from_dict(cls, d: dict) -> 'BtreeIndex':
        return cls()
