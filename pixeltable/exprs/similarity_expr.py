from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import sqlalchemy as sql
from typing_extensions import Self

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.catalog.globals import QColumnId
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
    idx_name: str | None = None  # index name; None if not specified by the user
    qcol_id: QColumnId | None = None  # identifies the indexed column

    def __init__(
        self,
        item: Expr,
        col_ref: ColumnRef | None = None,
        idx_name: str | None = None,
        qcol_id: QColumnId | None = None,
        table_version_key: TableVersionKey | None = None,
    ) -> None:
        from pixeltable.index import EmbeddingIndex

        super().__init__(ts.FloatType())
        self.components = [item]
        self.idx_name = idx_name

        # We store `qcol_id` + `table_version_key` instead of keeping `col_ref` as a component in
        # `self.components`. ColumnRef gets evaluated and materialized during query execution, which
        # is unnecessary here — we only need the column to resolve the embedding index at plan time,
        # not at eval time. Storing column and table version identifiers avoids that overhead.
        if col_ref is not None:
            tv = col_ref.tbl.get()
            column = col_ref.col
            self.qcol_id = column.qid
            self.table_version_key = tv.key
        else:
            # During deserialization
            assert table_version_key is not None
            assert qcol_id is not None
            self.qcol_id = qcol_id
            self.table_version_key = table_version_key
            tv = get_runtime().catalog.get_tbl_version(
                self.table_version_key, check_pending_ops=False, validate_initialized=False
            )
            column = tv.path.get_column_by_qid(self.qcol_id)
            if column is None:
                raise excs.Error(
                    f'Column {self.qcol_id!r} not found in table version {self.table_version_key!r} or its bases'
                )
        # Get embedding index for given column
        idx_info = tv.get_idx(column, self.idx_name, EmbeddingIndex)
        idx = idx_info.idx
        assert isinstance(idx, EmbeddingIndex)
        # Ensure we always store the concrete index name resolved by the catalog, even if idx_name was omitted
        self.idx_name = idx_info.name

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
        assert self.qcol_id is not None

        tbl_version = get_runtime().catalog.get_tbl_version(self.table_version_key, validate_initialized=True)
        col = tbl_version.path.get_column_by_qid(self.qcol_id)
        if col is None:
            return f'<invalid>.similarity({self.components[0]}, {self.idx_name!r})'
        return f'{col.name}.similarity({self.components[0]}, {self.idx_name!r})'

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [
            *super()._id_attrs(),
            ('table_version_key', self.table_version_key),
            ('qcol_id', self.qcol_id),
            ('idx_name', self.idx_name),
        ]

    def _equals(self, other: 'SimilarityExpr') -> bool:
        return (
            self.table_version_key == other.table_version_key
            and self.qcol_id == other.qcol_id
            and self.idx_name == other.idx_name
        )

    def tbl_ids(self) -> set[UUID]:
        return {self.table_version_key.tbl_id} | super().tbl_ids()

    @classmethod
    def get_refd_column_ids(cls, expr_dict: dict[str, Any]) -> set[catalog.QColumnId]:
        result = super().get_refd_column_ids(expr_dict)
        if 'qcol_id' in expr_dict:
            result.add(
                catalog.QColumnId(tbl_id=UUID(expr_dict['qcol_id']['tbl_id']), col_id=expr_dict['qcol_id']['col_id'])
            )
        return result

    @property
    def validation_error(self) -> str | None:
        try:
            self._resolve_idx(validate_initialized=False)
            return None
        except excs.Error as e:
            return str(e)

    def is_bound_by(self, tbls: list[catalog.TableVersionPath]) -> bool:
        tbl_version = get_runtime().catalog.get_tbl_version(self.table_version_key, validate_initialized=True)
        col = tbl_version.path.get_column_by_qid(self.qcol_id)
        if col is None:
            return False
        return any(tbl.has_column(col) for tbl in tbls)

    def _retarget(self, tbl_versions: dict[UUID, catalog.TableVersion]) -> Self:
        super()._retarget(tbl_versions)
        tv = tbl_versions.get(self.table_version_key.tbl_id)
        if tv is not None:
            self.table_version_key = tv.key
        return self

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

    def _resolve_idx(self, validate_initialized: bool = True) -> 'TableVersion.IndexInfo':
        """Resolve the embedding index; validate_initialized=False is used during schema initialization."""
        from pixeltable.index import EmbeddingIndex

        tbl_version = get_runtime().catalog.get_tbl_version(
            self.table_version_key, validate_initialized=validate_initialized
        )
        col = tbl_version.path.get_column_by_qid(self.qcol_id)
        if col is None:
            raise excs.Error(
                f'Embedding index {self.idx_name!r} no longer exists because the indexed column was dropped'
            )
        # get_idx() raises if the index no longer exists (e.g. it was dropped)
        idx_info = tbl_version.get_idx(col, self.idx_name, EmbeddingIndex)
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise excs.Error('similarity(): cannot be used in a computed column')

    def _as_dict(self) -> dict:
        return {
            'idx_name': self.idx_name,
            'table_version_key': self.table_version_key.as_dict(),
            'qcol_id': {'tbl_id': str(self.qcol_id.tbl_id), 'col_id': self.qcol_id.col_id},
            **super()._as_dict(),
        }

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> 'SimilarityExpr':
        table_version_key = TableVersionKey.from_dict(d['table_version_key'])
        idx_name = d.get('idx_name')
        qcol_id = QColumnId(tbl_id=UUID(d['qcol_id']['tbl_id']), col_id=d['qcol_id']['col_id'])
        return cls(item=components[0], idx_name=idx_name, table_version_key=table_version_key, qcol_id=qcol_id)
