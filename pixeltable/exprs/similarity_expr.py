from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np
import sqlalchemy as sql
from typing_extensions import Self

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.catalog.globals import QColumnId
from pixeltable.catalog.table_version import TableVersionKey
from pixeltable.env import Env
from ..catalog import ColumnVersionMd

from ..runtime import get_runtime
from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .literal import Literal
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache
from .variable import Variable

if TYPE_CHECKING:
    from pixeltable.catalog.table_version import TableVersion


class SimilarityExpr(Expr):
    """
    A similarity expression against an embedding index.
    """

    table_version_key: TableVersionKey
    idx_name: str | None = None  # index name; None if not specified by the user
    qcol_id: QColumnId | None = None  # identifies the indexed column
    _embedding: np.ndarray | None = None  # populated by prepare()

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
        col_md: ColumnVersionMd
        if col_ref is not None:
            #tv = col_ref.tbl_version.get()
            col_md = col_ref.col_md
            self.qcol_id = col_md.qcolid
            self.table_version_key = TableVersionKey(col_md.tbl_id, col_md.effective_version)
        else:
            # During deserialization
            assert table_version_key is not None
            assert qcol_id is not None
            self.qcol_id = qcol_id
            self.table_version_key = table_version_key
            tbl_uri = Env.get().tbl_catalog_uri(table_version_key.tbl_id)
            tv = get_runtime().catalog.get_tbl_version(self.table_version_key, validate_initialized=False)
            # Find column in the table hierarchy (tv could be a snapshot)
            column = tv.lookup_column(self.qcol_id)
            if column is None:
                raise excs.NotFoundError(
                    excs.ErrorCode.COLUMN_NOT_FOUND,
                    f'Column {self.qcol_id!r} not found in table version {self.table_version_key!r} or its bases',
                )

        # Get embedding index for given column
        if tv.supports_idxs:
            idx_info = tv.get_idx(column, self.idx_name, EmbeddingIndex)
            idx = idx_info.idx
            assert isinstance(idx, EmbeddingIndex)
            # Ensure we always store the concrete index name resolved by the catalog, even if idx_name was omitted
            self.idx_name = idx_info.name

            # Skip for array columns; similarity search uses the raw vector directly.
            if not item.col_type.is_array_type() and item.col_type._type not in idx.embeddings:
                type_str = item.col_type._type.name.lower()
                article = 'an' if type_str[0] in 'aeiou' else 'a'
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Embedding index {idx_info.name!r} on column {idx_info.col.name!r} does not have {article} '
                    f'{type_str} embedding and does not support {type_str} queries',
                )
        self.id = self._create_id()

    def __repr__(self) -> str:
        idx_str = f'{self.idx_name!r}' if self.idx_name is not None else '<no index>'
        assert self.qcol_id is not None

        tbl_version = get_runtime().catalog.get_tbl_version(self.table_version_key, validate_initialized=True)
        col = tbl_version.path.get_column_by_id(self.qcol_id) if tbl_version.path is not None else None
        if col is None:
            return f'<invalid>.similarity({self.components[0]}, {idx_str})'
        return f'{col.name}.similarity({self.components[0]}, {idx_str})'

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
        tbl_version = get_runtime().catalog.get_tbl_version(self.table_version_key, validate_initialized=False)
        if not tbl_version.supports_idxs:
            # nothing to validate
            return None
        try:
            self._resolve_idx(validate_initialized=False)
            return None
        except excs.Error as e:
            return str(e)

    def is_bound_by(self, tbls: list[catalog.TablePath]) -> bool:
        tbl_version = get_runtime().catalog.get_tbl_version(self.table_version_key, validate_initialized=True)
        col = tbl_version.lookup_column(self.qcol_id)
        if col is None:
            return False
        return any(tbl.has_column(col.qid) for tbl in tbls)

    def _retarget(self, tbl_versions: dict[UUID, catalog.TableVersion]) -> Self:
        super()._retarget(tbl_versions)
        tv = tbl_versions.get(self.table_version_key.tbl_id)
        if tv is not None:
            self.table_version_key = tv.key
        return self

    def default_column_name(self) -> str:
        return 'similarity'

    @property
    def _bind_name(self) -> str:
        assert self.id is not None
        # 'pxt_system:' for system binds, 'pxt:<identifier>' for user Variable binds: keep the two namespaces disjoint
        return f'pxt_system:sim_{self.id}'

    def prepare(self, args: dict[str, Any], bound_args: dict[str, Any]) -> None:
        super().prepare(args, bound_args)
        # Only the Variable path needs to register a value: sql_expr() emits a bindparam in that
        # case, and we resolve the embedding from the bound Variable here. The Literal path inlines
        # the embedding directly in sql_expr() and never reads from bound_args.
        item = self.components[0]
        if isinstance(item, Variable):
            from pixeltable.index import EmbeddingIndex

            idx_info = self._resolve_idx()
            assert isinstance(idx_info.idx, EmbeddingIndex)
            self._embedding = idx_info.idx.compute_query_embedding(
                item._bound_val, item.col_type, idx_info.val_col.col_type
            )
            bound_args[self._bind_name] = self._embedding

    def _query_element(self) -> sql.ColumnElement:
        """Build the query operand for the similarity SQL clause.

        For Literals, compute the embedding now and inline it into the SQL. For Variables, emit a
        bindparam whose value is supplied by bound_args at execute time (populated in prepare()).
        """
        from pixeltable.index import EmbeddingIndex

        idx_info = self._resolve_idx()
        assert isinstance(idx_info.idx, EmbeddingIndex)
        item = self.components[0]
        if isinstance(item, Literal):
            embedding = idx_info.idx.compute_query_embedding(item.val, item.col_type, idx_info.val_col.col_type)
            return sql.literal(embedding, type_=idx_info.val_col.sa_col.type)
        if isinstance(item, Variable):
            return sql.bindparam(self._bind_name, type_=idx_info.val_col.sa_col.type)
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION, 'similarity(): requires a value, not an expression'
        )

    def sql_expr(self, _: SqlElementCache) -> sql.ColumnElement | None:
        from pixeltable.index import EmbeddingIndex

        idx_info = self._resolve_idx()
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info.idx.similarity_clause(idx_info.val_col, self._query_element())

    def as_order_by_clause(self, is_asc: bool) -> sql.ColumnElement | None:
        from pixeltable.index import EmbeddingIndex

        idx_info = self._resolve_idx()
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info.idx.order_by_clause(idx_info.val_col, self._query_element(), is_asc)

    def _resolve_idx(self, validate_initialized: bool = True) -> 'TableVersion.IndexInfo':
        """Resolve the embedding index; validate_initialized=False is used during schema initialization."""
        from pixeltable.index import EmbeddingIndex

        tbl_version = get_runtime().catalog.get_tbl_version(
            self.table_version_key, validate_initialized=validate_initialized
        )
        col = tbl_version.lookup_column(self.qcol_id)
        if col is None:
            raise excs.NotFoundError(
                excs.ErrorCode.INDEX_NOT_FOUND,
                f'Embedding index {self.idx_name!r} no longer exists because the indexed column was dropped',
            )
        # get_idx() raises if the index no longer exists (e.g. it was dropped)
        idx_info = tbl_version.get_idx(col, self.idx_name, EmbeddingIndex)
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION, 'similarity(): cannot be used in a computed column'
        )

    def _as_dict(self) -> dict:
        return {
            'idx_name': self.idx_name,
            'table_version_key': self.table_version_key.as_dict(),
            'qcol_id': {'tbl_id': str(self.qcol_id.tbl_id), 'col_id': self.qcol_id.col_id},
            **super()._as_dict(),
        }

    @classmethod
    def _from_dict(
        cls, d: dict, components: list[Expr], tbl_versions: dict[UUID, catalog.TableVersion] | None = None
    ) -> 'SimilarityExpr':
        tvk_from_dict = TableVersionKey.from_dict(d['table_version_key'])
        if tbl_versions is not None:
            # Ignore table version key from the dict, retarget to the provided table version instead
            table_version_key = tbl_versions[tvk_from_dict.tbl_id].key
        else:
            table_version_key = tvk_from_dict
        idx_name = d.get('idx_name')
        qcol_id = QColumnId(tbl_id=UUID(d['qcol_id']['tbl_id']), col_id=d['qcol_id']['col_id'])
        return cls(item=components[0], idx_name=idx_name, table_version_key=table_version_key, qcol_id=qcol_id)
