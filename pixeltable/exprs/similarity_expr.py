from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import catalog
from pixeltable.catalog.table_version import TableVersionKey

from ..runtime import get_runtime
from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .literal import Literal
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache

if TYPE_CHECKING:
    from pixeltable.catalog.table_version import TableVersion


class SimilarityExpr(Expr):
    """
    A similarity expression against an embedding index.
    """

    table_version_key: TableVersionKey
    idx_name: str | None = None  # index name could be optional
    col_id: int | None = None  # used to find the first embedding index when index name is not provided

    def __init__(
        self,
        item: Expr,
        col_ref: ColumnRef | None = None,
        idx_name: str | None = None,
        col_id: int | None = None,
        table_version_key: TableVersionKey | None = None,
    ) -> None:
        from pixeltable.index import EmbeddingIndex

        super().__init__(ts.FloatType())
        self.components = [item]
        self.col_id = col_id
        self.table_version_key = table_version_key
        self.idx_name = idx_name

        if col_ref is not None:
            tv = col_ref.tbl.get()
            column = col_ref.col
            self.col_id = column.id
            self.table_version_key = tv.key
        else:
            # During deserialization
            assert self.table_version_key is not None
            assert self.col_id is not None
            tv = get_runtime().catalog.get_tbl_version(
                self.table_version_key, check_pending_ops=False, validate_initialized=False
            )
            column = tv.cols_by_id[self.col_id]
            assert column is not None

        # determine index to use
        if self.idx_name is None:
            # Look up index by column
            idx_info = tv.get_idx(column, None, EmbeddingIndex)
            self.idx_name = idx_info.name
        else:
            # Look up index by name, not by id, so that the expr is still valid when the index
            # gets dropped and re-created
            idx_info = tv.idxs_by_name.get(idx_name)
            if idx_info is None:
                raise excs.Error(f'Index {idx_name!r} not found for column {column.name!r}')

        idx = idx_info.idx
        assert isinstance(idx, EmbeddingIndex)

        # Skip for array columns; similarity search uses the raw vector directly.
        if not item.col_type.is_array_type() and item.col_type._type not in idx.embeddings:
            type_str = item.col_type._type.name.lower()
            article = 'an' if type_str[0] in 'aeiou' else 'a'
            raise excs.Error(
                f'Embedding index {idx_info.name!r} on column {idx_info.col.name!r} does not have {article} '
                f'{type_str} embedding and does not support {type_str} queries'
            )
        self.id = self._create_id()

    def __repr__(self) -> str:
        assert self.idx_name is not None
        assert self.col_id is not None

        tbl_version = get_runtime().catalog.get_tbl_version(self.table_version_key, validate_initialized=True)
        col = tbl_version.cols_by_id[self.col_id]
        return f'{col.name}.similarity({self.components[0]}, {self.idx_name!r})'

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('idx_name', self.idx_name)]

    def tbl_ids(self) -> set[UUID]:
        return {self.table_version_key.tbl_id} | super().tbl_ids()

    def is_bound_by(self, tbls: list[catalog.TableVersionPath]) -> bool:
        tbl_version = get_runtime().catalog.get_tbl_version(self.table_version_key, validate_initialized=True)
        col = tbl_version.cols_by_id[self.col_id]
        return any(tbl.has_column(col) for tbl in tbls)

    def default_column_name(self) -> str:
        return 'similarity'

    def sql_expr(self, _: SqlElementCache) -> sql.ColumnElement | None:
        from pixeltable.index import EmbeddingIndex

        # check for a literal here, instead of the c'tor: needed for ExprTemplateFunctions
        if not isinstance(self.components[0], Literal):
            raise excs.Error('similarity(): requires a value, not an expression')
        idx_info = self._resolve_idx()
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info.idx.similarity_clause(idx_info.val_col, self.components[0])

    def as_order_by_clause(self, is_asc: bool) -> sql.ColumnElement | None:
        from pixeltable.index import EmbeddingIndex

        # check for a literal here, instead of the c'tor: needed for ExprTemplateFunctions
        if not isinstance(self.components[0], Literal):
            raise excs.Error('similarity(): requires a value, not an expression')
        idx_info = self._resolve_idx()
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info.idx.order_by_clause(idx_info.val_col, self.components[0], is_asc)

    def _resolve_idx(self) -> 'TableVersion.IndexInfo':
        from pixeltable.index import EmbeddingIndex

        tbl_version = get_runtime().catalog.get_tbl_version(self.table_version_key, validate_initialized=True)
        idx_info = tbl_version.idxs_by_name.get(self.idx_name)
        if idx_info is None:
            col = tbl_version.cols_by_id[self.col_id]
            raise excs.Error(
                f'Embedding index {self.idx_name!r} on column {col.name!r} no longer exists (it may have been dropped)'
            )
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise excs.Error('similarity(): cannot be used in a computed column')

    def _as_dict(self) -> dict:
        return {
            'idx_name': self.idx_name,
            'table_version_key': self.table_version_key.as_dict(),
            'col_id': self.col_id,
            **super()._as_dict(),
        }

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> 'SimilarityExpr':
        table_version_key = TableVersionKey.from_dict(d['table_version_key'])
        idx_name = d.get('idx_name')
        col_id = d.get('col_id')
        return cls(item=components[0], idx_name=idx_name, table_version_key=table_version_key, col_id=col_id)
