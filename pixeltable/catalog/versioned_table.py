from __future__ import annotations

import dataclasses
import importlib
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Optional
from uuid import UUID

import jsonschema.exceptions
import sqlalchemy as sql
import sqlalchemy.orm as orm

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
import pixeltable.index as index
import pixeltable.type_system as ts
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator
from pixeltable.metadata import schema
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.media_store import MediaStore

from ..func.globals import resolve_symbol
from .column import Column
from .globals import _POS_COLUMN_NAME, _ROWID_COLUMN_NAME, MediaValidation, UpdateStatus, is_valid_identifier
from .table_version import TableVersion

if TYPE_CHECKING:
    from pixeltable import exec, store

_logger = logging.getLogger('pixeltable')


class VersionedTable:
    """
    Contains the non-versioned metadata of a single physical table. An instance corresponds to a single row in
    schema.Tables and its corresponding data table in the store.

    Every TableVersion instance belongs to a single VersionedTable instance.
    """
    id: UUID
    name: str
    base: Optional[VersionedTable]
    view_md: Optional[schema.ViewMd]
    predicate: Optional[exprs.Expr]
    iterator_cls: Optional[type[ComponentIterator]]
    iterator_args: Optional[exprs.InlineDict]
    num_iterator_cols: int

    # contains complete history of columns, incl dropped ones
    cols: list[Column]
    # needed for _create_tbl_md()
    idx_md: dict[int, schema.IndexMd]
    # contains only actively maintained indices
    idxs_by_name: dict[str, VersionedTable.IndexInfo]

    external_stores: dict[str, pxt.io.ExternalStore]
    store_tbl: 'store.StoreBase'

    @dataclasses.dataclass
    class IndexInfo:
        id: int
        name: str
        idx: index.IndexBase
        col: Column
        val_col: Column
        undo_col: Column

    def __init__(
            self,
            id: UUID,
            tbl_md: schema.TableMd,
            base: Optional[VersionedTable] = None,
    ):
        self.id = id
        self.name = tbl_md.name
        self.base = base
        self.view_md = tbl_md.view_md  # save this as-is, it's needed for _create_md()
        is_view = tbl_md.view_md is not None

        # view-specific initialization
        from pixeltable import exprs

        predicate_dict = None if not is_view or tbl_md.view_md.predicate is None else tbl_md.view_md.predicate
        self.predicate = exprs.Expr.from_dict(predicate_dict) if predicate_dict is not None else None

        # component view-specific initialization
        self.iterator_cls = None
        self.iterator_args = None
        self.num_iterator_cols = 0
        if is_view and tbl_md.view_md.iterator_class_fqn is not None:
            module_name, class_name = tbl_md.view_md.iterator_class_fqn.rsplit('.', 1)
            module = importlib.import_module(module_name)
            self.iterator_cls = getattr(module, class_name)
            self.iterator_args = exprs.InlineDict.from_dict(tbl_md.view_md.iterator_args)
            output_schema, _ = self.iterator_cls.output_schema(**self.iterator_args.to_kwargs())
            self.num_iterator_cols = len(output_schema)
            assert tbl_md.view_md.iterator_args is not None

        # init schema after we determined whether we're a component view, and before we create the store table
        self.cols = []
        self.idx_md = tbl_md.index_md
        self.idxs_by_name = {}
        self.external_stores = {}

        self._init_schema(tbl_md)

        # Init external stores (this needs to happen after the schema is created)
        self._init_external_stores(tbl_md)

    @property
    def is_snapshot(self) -> bool:
        return self.view_md is not None and self.view_md.is_snapshot

    def _init_schema(self, tbl_md: schema.TableMd, schema_version_md: schema.TableSchemaVersionMd) -> None:
        # create columns first, so the indices can reference them
        self._init_cols(tbl_md, schema_version_md)
        self._init_idxs(tbl_md)
        # create the sa schema only after creating the columns and indices
        self._init_sa_schema()

    def _init_cols(self, tbl_md: schema.TableMd) -> None:
        self.cols = []
        for col_md in tbl_md.column_md.values():
            schema_col_md = schema_version_md.columns[col_md.id] if col_md.id in schema_version_md.columns else None
            col_name = schema_col_md.name if schema_col_md is not None else None
            media_val = (
                MediaValidation[schema_col_md.media_validation.upper()]
                if schema_col_md is not None and schema_col_md.media_validation is not None
                else None
            )
            col = Column(
                col_id=col_md.id,
                name=col_name,
                col_type=ts.ColumnType.from_dict(col_md.col_type),
                is_pk=col_md.is_pk,
                stored=col_md.stored,
                media_validation=media_val,
                schema_version_add=col_md.schema_version_add,
                schema_version_drop=col_md.schema_version_drop,
                value_expr_dict=col_md.value_expr,
            )
            col.tbl = self
            self.cols.append(col)

            # populate the lookup structures before Expr.from_dict()
            if col_md.schema_version_add > self.schema_version:
                # column was added after this version
                continue
            if col_md.schema_version_drop is not None and col_md.schema_version_drop <= self.schema_version:
                # column was dropped
                continue
            if col.name is not None:
                self.cols_by_name[col.name] = col
            self.cols_by_id[col.id] = col

            # make sure to traverse columns ordered by position = order in which cols were created;
            # this guarantees that references always point backwards
            if col_md.value_expr is not None:
                refd_cols = exprs.Expr.get_refd_columns(col_md.value_expr)
                self._record_refd_columns(col)

    def _init_idxs(self, tbl_md: schema.TableMd) -> None:
        self.idx_md = tbl_md.index_md
        self.idxs_by_name = {}
        import pixeltable.index as index_module

        for md in tbl_md.index_md.values():
            if md.schema_version_add > self.schema_version or (
                    md.schema_version_drop is not None and md.schema_version_drop <= self.schema_version
            ):
                # index not visible in this schema version
                continue

            # instantiate index object
            cls_name = md.class_fqn.rsplit('.', 1)[-1]
            cls = getattr(index_module, cls_name)
            idx_col = self._get_column(UUID(md.indexed_col_tbl_id), md.indexed_col_id)
            idx = cls.from_dict(idx_col, md.init_args)

            # fix up the sa column type of the index value and undo columns
            val_col = self.cols_by_id[md.index_val_col_id]
            val_col.sa_col_type = idx.index_sa_type()
            val_col._records_errors = False
            undo_col = self.cols_by_id[md.index_val_undo_col_id]
            undo_col.sa_col_type = idx.index_sa_type()
            undo_col._records_errors = False
            idx_info = self.IndexInfo(id=md.id, name=md.name, idx=idx, col=idx_col, val_col=val_col,
                                      undo_col=undo_col)
            self.idxs_by_name[md.name] = idx_info

    def _init_sa_schema(self) -> None:
        # create the sqlalchemy schema; do this after instantiating columns, in order to determine whether they
        # need to record errors
        from pixeltable.store import StoreComponentView, StoreTable, StoreView

        if self.is_component_view():
            self.store_tbl = StoreComponentView(self)
        elif self.is_view():
            self.store_tbl = StoreView(self)
        else:
            self.store_tbl = StoreTable(self)

