from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
import time
from typing import Optional, List, Dict, Any, Tuple, Type, Set, Iterable
import uuid
from uuid import UUID

import sqlalchemy as sql
import sqlalchemy.orm as orm

import pixeltable
import pixeltable.func as func
import pixeltable.type_system as ts
import pixeltable.exceptions as excs
import pixeltable.index as index
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator
from pixeltable.metadata import schema
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.media_store import MediaStore
from .column import Column
from .globals import UpdateStatus, POS_COLUMN_NAME, is_valid_identifier
from ..func.globals import resolve_symbol

_logger = logging.getLogger('pixeltable')

class TableVersion:
    """
    TableVersion represents a particular version of a table/view along with its physical representation:
    - the physical representation is a store table with indices
    - the version can be mutable or a snapshot
    - tables and their recursive views form a tree, and a mutable TableVersion also records its own
      mutable views in order to propagate updates
    - each view TableVersion records its base:
      * the base is correct only for mutable views (snapshot versions form a DAG, not a tree)
      * the base is useful for getting access to the StoreTable and the base id
      * TODO: create a separate hierarchy of objects that records the version-independent tree of tables/views, and
        have TableVersions reference those
    - mutable TableVersions record their TableVersionPath, which is needed for expr evaluation in updates
    """
    @dataclasses.dataclass
    class IndexInfo:
        id: int
        name: str
        idx: index.IndexBase
        col: Column
        val_col: Column
        undo_col: Column


    def __init__(
            self, id: UUID, tbl_md: schema.TableMd, version: int, schema_version_md: schema.TableSchemaVersionMd,
            base: Optional[TableVersion] = None, base_path: Optional['pixeltable.catalog.TableVersionPath'] = None,
            is_snapshot: Optional[bool] = None
    ):
        # only one of base and base_path can be non-None
        assert base is None or base_path is None
        self.id = id
        self.name = tbl_md.name
        self.version = version
        self.comment = schema_version_md.comment
        self.num_retained_versions = schema_version_md.num_retained_versions
        self.schema_version = schema_version_md.schema_version
        self.view_md = tbl_md.view_md  # save this as-is, it's needed for _create_md()
        is_view = tbl_md.view_md is not None
        self.is_snapshot = (is_view and tbl_md.view_md.is_snapshot) or bool(is_snapshot)
        # a mutable TableVersion doesn't have a static version
        self.effective_version = self.version if self.is_snapshot else None

        # mutable tables need their TableVersionPath for expr eval during updates
        from .table_version_path import TableVersionPath
        if self.is_snapshot:
            self.path = None
        else:
            self.path = TableVersionPath(self, base=base_path) if base_path is not None else TableVersionPath(self)

        self.base = base_path.tbl_version if base_path is not None else base
        if self.is_snapshot:
            self.next_col_id = -1
            self.next_idx_id = -1  # TODO: can snapshots have separate indices?
            self.next_rowid = -1
        else:
            assert tbl_md.current_version == self.version
            self.next_col_id = tbl_md.next_col_id
            self.next_idx_id = tbl_md.next_idx_id
            self.next_rowid = tbl_md.next_row_id

        self.remotes = dict(TableVersion._init_remote(remote_md) for remote_md in tbl_md.remotes)

        # view-specific initialization
        from pixeltable import exprs
        predicate_dict = None if not is_view or tbl_md.view_md.predicate is None else tbl_md.view_md.predicate
        self.predicate = exprs.Expr.from_dict(predicate_dict) if predicate_dict is not None else None
        self.mutable_views: List[TableVersion] = []  # targets for update propagation
        if self.base is not None and not self.base.is_snapshot and not self.is_snapshot:
            self.base.mutable_views.append(self)

        # component view-specific initialization
        self.iterator_cls: Optional[Type[ComponentIterator]] = None
        self.iterator_args: Optional[exprs.InlineDict] = None
        self.num_iterator_cols = 0
        if is_view and tbl_md.view_md.iterator_class_fqn is not None:
            module_name, class_name = tbl_md.view_md.iterator_class_fqn.rsplit('.', 1)
            module = importlib.import_module(module_name)
            self.iterator_cls = getattr(module, class_name)
            self.iterator_args = exprs.Expr.from_dict(tbl_md.view_md.iterator_args)
            assert isinstance(self.iterator_args, exprs.InlineDict)
            output_schema, _ = self.iterator_cls.output_schema(**self.iterator_args.to_dict())
            self.num_iterator_cols = len(output_schema)
            assert tbl_md.view_md.iterator_args is not None

        # register this table version now so that it's available when we're re-creating value exprs
        import pixeltable.catalog as catalog
        cat = catalog.Catalog.get()
        cat.tbl_versions[(self.id, self.effective_version)] = self

        # init schema after we determined whether we're a component view, and before we create the store table
        self.cols: list[Column] = []  # contains complete history of columns, incl dropped ones
        self.cols_by_name: dict[str, Column] = {}  # contains only user-facing (named) columns visible in this version
        self.cols_by_id: dict[int, Column] = {}  # contains only columns visible in this version, both system and user
        self.idx_md = tbl_md.index_md  # needed for _create_tbl_md()
        self.idxs_by_name: dict[str, TableVersion.IndexInfo] = {}  # contains only actively maintained indices
        self._init_schema(tbl_md, schema_version_md)

    def __hash__(self) -> int:
        return hash(self.id)

    def create_snapshot_copy(self) -> TableVersion:
        """Create a snapshot copy of this TableVersion"""
        assert not self.is_snapshot
        return TableVersion(
            self.id, self._create_tbl_md(), self.version,
            self._create_schema_version_md(preceding_schema_version=0),  # preceding_schema_version: dummy value
            is_snapshot=True, base=self.base)

    @classmethod
    def create(
            cls, session: orm.Session, dir_id: UUID, name: str, cols: List[Column], num_retained_versions: int,
            comment: str, base_path: Optional['pixeltable.catalog.TableVersionPath'] = None,
            view_md: Optional[schema.ViewMd] = None
    ) -> Tuple[UUID, Optional[TableVersion]]:
        # assign ids
        cols_by_name: Dict[str, Column] = {}
        for pos, col in enumerate(cols):
            col.id = pos
            col.schema_version_add = 0
            cols_by_name[col.name] = col
            if col.value_expr is None and col.compute_func is not None:
                cls._create_value_expr(col, base_path)
            if col.is_computed:
                col.check_value_expr()

        timestamp = time.time()
        # create schema.Table
        # Column.dependent_cols for existing cols is wrong at this point, but init() will set it correctly
        column_md = cls._create_column_md(cols)
        table_md = schema.TableMd(
            name=name, current_version=0, current_schema_version=0, next_col_id=len(cols),
            next_idx_id=0, next_row_id=0, column_md=column_md, index_md={}, remotes=[], view_md=view_md)
        # create a schema.Table here, we need it to call our c'tor;
        # don't add it to the session yet, we might add index metadata
        tbl_id = uuid.uuid4()
        tbl_record = schema.Table(id=tbl_id, dir_id=dir_id, md=dataclasses.asdict(table_md))

        # create schema.TableVersion
        table_version_md = schema.TableVersionMd(created_at=timestamp, version=0, schema_version=0)
        tbl_version_record = schema.TableVersion(
            tbl_id=tbl_record.id, version=0, md=dataclasses.asdict(table_version_md))

        # create schema.TableSchemaVersion
        schema_col_md = {col.id: schema.SchemaColumn(pos=pos, name=col.name) for pos, col in enumerate(cols)}

        schema_version_md = schema.TableSchemaVersionMd(
            schema_version=0, preceding_schema_version=None, columns=schema_col_md,
            num_retained_versions=num_retained_versions, comment=comment)
        schema_version_record = schema.TableSchemaVersion(
            tbl_id=tbl_record.id, schema_version=0, md=dataclasses.asdict(schema_version_md))

        # if this is purely a snapshot (it doesn't require any additional storage for columns and it # doesn't have a
        # predicate to apply at runtime), we don't create a physical table and simply use the base's table version path
        if view_md is not None and view_md.is_snapshot and view_md.predicate is None and len(cols) == 0:
            session.add(tbl_record)
            session.add(tbl_version_record)
            session.add(schema_version_record)
            return tbl_record.id, None

        assert (base_path is not None) == (view_md is not None)
        base = base_path.tbl_version if base_path is not None and view_md.is_snapshot else None
        base_path = base_path if base_path is not None and not view_md.is_snapshot else None
        tbl_version = cls(tbl_record.id, table_md, 0, schema_version_md, base=base, base_path=base_path)

        conn = session.connection()
        tbl_version.store_tbl.create(conn)
        if view_md is None or not view_md.is_snapshot:
            # add default indices, after creating the store table
            for col in tbl_version.cols_by_name.values():
                status = tbl_version._add_default_index(col, conn=conn)
                assert status is None or status.num_excs == 0

        # we re-create the tbl_record here, now that we have new index metadata
        tbl_record = schema.Table(id=tbl_id, dir_id=dir_id, md=dataclasses.asdict(tbl_version._create_tbl_md()))
        session.add(tbl_record)
        session.add(tbl_version_record)
        session.add(schema_version_record)
        return tbl_record.id, tbl_version

    @classmethod
    def delete_md(cls, tbl_id: UUID, conn: sql.Connection) -> None:
        conn.execute(
            sql.delete(schema.TableSchemaVersion.__table__).where(schema.TableSchemaVersion.tbl_id == tbl_id))
        conn.execute(
            sql.delete(schema.TableVersion.__table__).where(schema.TableVersion.tbl_id == tbl_id))
        conn.execute(sql.delete(schema.Table.__table__).where(schema.Table.id == tbl_id))

    def drop(self) -> None:
        with Env.get().engine.begin() as conn:
            # delete this table and all associated data
            MediaStore.delete(self.id)
            FileCache.get().clear(tbl_id=self.id)
            self.delete_md(self.id, conn)
            self.store_tbl.drop(conn)

        # de-register table version from catalog
        from .catalog import Catalog
        cat = Catalog.get()
        del cat.tbl_versions[(self.id, self.effective_version)]
        # TODO: remove from tbl_dependents

    def _init_schema(self, tbl_md: schema.TableMd, schema_version_md: schema.TableSchemaVersionMd) -> None:
        # create columns first, so the indices can reference them
        self._init_cols(tbl_md, schema_version_md)
        self._init_idxs(tbl_md)
        # create the sa schema only after creating the columns and indices
        self._init_sa_schema()

    def _init_cols(self, tbl_md: schema.TableMd, schema_version_md: schema.TableSchemaVersionMd) -> None:
        """Initialize self.cols with the columns visible in our effective version"""
        import pixeltable.exprs as exprs
        self.cols = []
        self.cols_by_name = {}
        self.cols_by_id = {}
        for col_md in tbl_md.column_md.values():
            col_name = schema_version_md.columns[col_md.id].name if col_md.id in schema_version_md.columns else None
            col = Column(
                col_id=col_md.id, name=col_name, col_type=ts.ColumnType.from_dict(col_md.col_type),
                is_pk=col_md.is_pk, stored=col_md.stored,
                schema_version_add=col_md.schema_version_add, schema_version_drop=col_md.schema_version_drop)
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
                col.value_expr = exprs.Expr.from_dict(col_md.value_expr)
                self._record_value_expr(col)

            # if this is a stored proxy column, resolve the relationships with its proxy base.
            if col_md.proxy_base is not None:
                # proxy_base must have a strictly smaller id, so we must already have encountered it
                # in traversal order; and if the proxy column is active at this version, then the
                # proxy base must necessarily be active as well. This motivates the following assertion.
                assert col_md.proxy_base in self.cols_by_id
                base_col = self.cols_by_id[col_md.proxy_base]
                base_col.stored_proxy = col
                col.proxy_base = base_col

    def _init_idxs(self, tbl_md: schema.TableMd) -> None:
        self.idx_md = tbl_md.index_md
        self.idxs_by_name = {}
        import pixeltable.index as index_module
        for md in tbl_md.index_md.values():
            if md.schema_version_add > self.schema_version \
                    or md.schema_version_drop is not None and md.schema_version_drop <= self.schema_version:
                # index not visible in this schema version
                continue

            # instantiate index object
            cls_name = md.class_fqn.rsplit('.', 1)[-1]
            cls = getattr(index_module, cls_name)
            idx_col = self.cols_by_id[md.indexed_col_id]
            idx = cls.from_dict(idx_col, md.init_args)

            # fix up the sa column type of the index value and undo columns
            val_col = self.cols_by_id[md.index_val_col_id]
            val_col.sa_col_type = idx.index_sa_type()
            val_col._records_errors = False
            undo_col = self.cols_by_id[md.index_val_undo_col_id]
            undo_col.sa_col_type = idx.index_sa_type()
            undo_col._records_errors = False
            idx_info = self.IndexInfo(id=md.id, name=md.name, idx=idx, col=idx_col, val_col=val_col, undo_col=undo_col)
            self.idxs_by_name[md.name] = idx_info

    def _init_sa_schema(self) -> None:
        # create the sqlalchemy schema; do this after instantiating columns, in order to determine whether they
        # need to record errors
        from pixeltable.store import StoreBase, StoreTable, StoreView, StoreComponentView
        if self.is_component_view():
            self.store_tbl: StoreBase = StoreComponentView(self)
        elif self.is_view():
            self.store_tbl: StoreBase = StoreView(self)
        else:
            self.store_tbl: StoreBase = StoreTable(self)

    def _update_md(
            self, timestamp: float, preceding_schema_version: Optional[int], conn: sql.engine.Connection
    ) -> None:
        """Update all recorded metadata in response to a data or schema change.
        Args:
            timestamp: timestamp of the change
            preceding_schema_version: last schema version if schema change, else None
        """
        conn.execute(
            sql.update(schema.Table.__table__)
                .values({schema.Table.md: dataclasses.asdict(self._create_tbl_md())})
                .where(schema.Table.id == self.id))

        version_md = self._create_version_md(timestamp)
        conn.execute(
            sql.insert(schema.TableVersion.__table__)
                .values(tbl_id=self.id, version=self.version, md=dataclasses.asdict(version_md)))
        if preceding_schema_version is not None:
            schema_version_md = self._create_schema_version_md(preceding_schema_version)
            conn.execute(
                sql.insert(schema.TableSchemaVersion.__table__)
                .values(
                    tbl_id=self.id, schema_version=self.schema_version,
                    md=dataclasses.asdict(schema_version_md)))

    def _store_idx_name(self, idx_id: int) -> str:
        """Return name of index in the store, which needs to be globally unique"""
        return f'idx_{self.id.hex}_{idx_id}'

    def add_index(self, col: Column, idx_name: Optional[str], idx: index.IndexBase) -> UpdateStatus:
        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version
        with Env.get().engine.begin() as conn:
            status = self._add_index(col, idx_name, idx, conn)
            self._update_md(time.time(), preceding_schema_version, conn)
            _logger.info(f'Added index {idx_name} on column {col.name} to table {self.name}')
            return status

    def _add_default_index(self, col: Column, conn: sql.engine.Connection) -> Optional[UpdateStatus]:
        """Add a B-tree index on this column if it has a compatible type"""
        if not col.stored:
            # if the column is intentionally not stored, we want to avoid the overhead of an index
            return None
        if not col.col_type.is_scalar_type() and not (col.col_type.is_media_type() and not col.is_computed):
            # wrong type for a B-tree
            return None
        if col.col_type.is_bool_type():
            # B-trees on bools aren't useful
            return None
        status = self._add_index(col, idx_name=None, idx=index.BtreeIndex(col), conn=conn)
        return status

    def _add_index(
            self, col: Column, idx_name: Optional[str], idx: index.IndexBase, conn: sql.engine.Connection
    ) -> UpdateStatus:
        assert not self.is_snapshot
        idx_id = self.next_idx_id
        self.next_idx_id += 1
        if idx_name is None:
            idx_name = f'idx{idx_id}'
        else:
            assert is_valid_identifier(idx_name)
            assert idx_name not in [i.name for i in self.idx_md.values()]

        # add the index value and undo columns (which need to be nullable)
        val_col = Column(
            col_id=self.next_col_id, name=None, computed_with=idx.index_value_expr(),
            sa_col_type=idx.index_sa_type(), stored=True,
            schema_version_add=self.schema_version, schema_version_drop=None,
            records_errors=idx.records_value_errors())
        val_col.tbl = self
        val_col.col_type = val_col.col_type.copy(nullable=True)
        self.next_col_id += 1

        undo_col = Column(
            col_id=self.next_col_id, name=None, col_type=val_col.col_type,
            sa_col_type=val_col.sa_col_type, stored=True,
            schema_version_add=self.schema_version, schema_version_drop=None,
            records_errors=False)
        undo_col.tbl = self
        undo_col.col_type = undo_col.col_type.copy(nullable=True)
        self.next_col_id += 1

        # create and register the index metadata
        idx_cls = type(idx)
        idx_md = schema.IndexMd(
            id=idx_id, name=idx_name,
            indexed_col_id=col.id, index_val_col_id=val_col.id, index_val_undo_col_id=undo_col.id,
            schema_version_add=self.schema_version, schema_version_drop=None,
            class_fqn=idx_cls.__module__ + '.' + idx_cls.__name__, init_args=idx.as_dict())
        idx_info = self.IndexInfo(id=idx_id, name=idx_name, idx=idx, col=col, val_col=val_col, undo_col=undo_col)
        self.idx_md[idx_id] = idx_md
        self.idxs_by_name[idx_name] = idx_info

        # add the columns and update the metadata
        status = self._add_columns([val_col, undo_col], conn)
        # now create the index structure
        idx.create_index(self._store_idx_name(idx_id), val_col, conn)

        return status

    def drop_index(self, idx_id: int) -> None:
        assert not self.is_snapshot
        assert idx_id in self.idx_md

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version
        idx_md = self.idx_md[idx_id]
        idx_md.schema_version_drop = self.schema_version
        assert idx_md.name in self.idxs_by_name
        idx_info = self.idxs_by_name[idx_md.name]
        del self.idxs_by_name[idx_md.name]

        with Env.get().engine.begin() as conn:
            self._drop_columns([idx_info.val_col, idx_info.undo_col])
            self._update_md(time.time(), preceding_schema_version, conn)
            _logger.info(f'Dropped index {idx_md.name} on table {self.name}')

    def add_column(self, col: Column, print_stats: bool = False) -> UpdateStatus:
        """Adds a column to the table.
        """
        assert not self.is_snapshot
        assert is_valid_identifier(col.name)
        assert col.stored is not None
        assert col.name not in self.cols_by_name
        col.tbl = self
        col.id = self.next_col_id
        self.next_col_id += 1

        if col.compute_func is not None:
            # create value_expr from compute_func
            self._create_value_expr(col, self.path)

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version
        with Env.get().engine.begin() as conn:
            status = self._add_columns([col], conn, print_stats=print_stats)
            _ = self._add_default_index(col, conn)
            # TODO: what to do about errors?
            self._update_md(time.time(), preceding_schema_version, conn)
        _logger.info(f'Added column {col.name} to table {self.name}, new version: {self.version}')

        msg = (
            f'Added {status.num_rows} column value{"" if status.num_rows == 1 else "s"} '
            f'with {status.num_excs} error{"" if status.num_excs == 1 else "s"}.'
        )
        print(msg)
        _logger.info(f'Column {col.name}: {msg}')
        return status

    def _add_columns(self, cols: List[Column], conn: sql.engine.Connection, print_stats: bool = False) -> UpdateStatus:
        """Add and populate columns within the current transaction"""
        row_count = self.store_tbl.count(conn=conn)
        for col in cols:
            if not col.col_type.nullable and not col.is_computed:
                if row_count > 0:
                    raise excs.Error(
                        f'Cannot add non-nullable column "{col.name}" to table {self.name} with existing rows')

        num_excs = 0
        cols_with_excs: List[Column] = []
        for col in cols:
            col.schema_version_add = self.schema_version
            # add the column to the lookup structures now, rather than after the store changes executed successfully,
            # because it might be referenced by the next column's value_expr
            self.cols.append(col)
            if col.name is not None:
                self.cols_by_name[col.name] = col
            self.cols_by_id[col.id] = col
            if col.value_expr is not None:
                col.check_value_expr()
                self._record_value_expr(col)

            if col.is_stored:
                self.store_tbl.add_column(col, conn)

            if not col.is_computed or not col.is_stored or row_count == 0:
                continue

            # populate the column
            from pixeltable.plan import Planner
            plan, value_expr_slot_idx = Planner.create_add_column_plan(self.path, col)
            plan.ctx.num_rows = row_count

            try:
                plan.ctx.conn = conn
                plan.open()
                num_excs = self.store_tbl.load_column(col, plan, value_expr_slot_idx, conn)
                if num_excs > 0:
                    cols_with_excs.append(col)
            except sql.exc.DBAPIError as e:
                self.cols.pop()
                for col in cols:
                    # remove columns that we already added
                    if col.id not in self.cols_by_id:
                        continue
                    if col.name is not None:
                        del self.cols_by_name[col.name]
                    del self.cols_by_id[col.id]
                # we need to re-initialize the sqlalchemy schema
                self.store_tbl.create_sa_tbl()
                raise excs.Error(f'Error during SQL execution:\n{e}')
            finally:
                plan.close()

        if print_stats:
            plan.ctx.profile.print(num_rows=row_count)
        # TODO(mkornacker): what to do about system columns with exceptions?
        return UpdateStatus(
            num_rows=row_count, num_computed_values=row_count, num_excs=num_excs,
            cols_with_excs=[f'{col.tbl.name}.{col.name}'for col in cols_with_excs if col.name is not None])

    def drop_column(self, name: str) -> None:
        """Drop a column from the table.
        """
        assert not self.is_snapshot
        if name not in self.cols_by_name:
            raise excs.Error(f'Unknown column: {name}')
        col = self.cols_by_name[name]
        dependent_user_cols = [c for c in col.dependent_cols if c.name is not None]
        if len(dependent_user_cols) > 0:
            raise excs.Error(
                f'Cannot drop column `{name}` because the following columns depend on it:\n'
                f'{", ".join(c.name for c in dependent_user_cols)}'
            )
        dependent_remotes = [remote for remote, col_mapping in self.remotes.items() if name in col_mapping]
        if len(dependent_remotes) > 0:
            raise excs.Error(
                f'Cannot drop column `{name}` because the following remotes depend on it:\n'
                f'{", ".join(str(r) for r in dependent_remotes)}'
            )
        assert col.stored_proxy is None  # since there are no dependent remotes

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        with Env.get().engine.begin() as conn:
            # drop this column and all dependent index columns and indices
            dropped_cols = [col]
            dropped_idx_names: List[str] = []
            for idx_info in self.idxs_by_name.values():
                if idx_info.col != col:
                    continue
                dropped_cols.extend([idx_info.val_col, idx_info.undo_col])
                idx_md = self.idx_md[idx_info.id]
                idx_md.schema_version_drop = self.schema_version
                assert idx_md.name in self.idxs_by_name
                dropped_idx_names.append(idx_md.name)
            # update idxs_by_name
            for idx_name in dropped_idx_names:
                del self.idxs_by_name[idx_name]
            self._drop_columns(dropped_cols)
            self._update_md(time.time(), preceding_schema_version, conn)
        _logger.info(f'Dropped column {name} from table {self.name}, new version: {self.version}')

    def _drop_columns(self, cols: list[Column]) -> None:
        """Mark columns as dropped"""
        assert not self.is_snapshot

        for col in cols:
            if col.value_expr is not None:
                # update Column.dependent_cols
                for c in self.cols:
                    if c == col:
                        break
                    c.dependent_cols.discard(col)

            col.schema_version_drop = self.schema_version
            if col.name is not None:
                assert col.name in self.cols_by_name
                del self.cols_by_name[col.name]
            assert col.id in self.cols_by_id
            del self.cols_by_id[col.id]

        self.store_tbl.create_sa_tbl()

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column.
        """
        assert not self.is_snapshot
        if old_name not in self.cols_by_name:
            raise excs.Error(f'Unknown column: {old_name}')
        if not is_valid_identifier(new_name):
            raise excs.Error(f"Invalid column name: '{new_name}'")
        if new_name in self.cols_by_name:
            raise excs.Error(f'Column {new_name} already exists')
        col = self.cols_by_name[old_name]
        del self.cols_by_name[old_name]
        col.name = new_name
        self.cols_by_name[new_name] = col

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        with Env.get().engine.begin() as conn:
            self._update_md(time.time(), preceding_schema_version, conn)
        _logger.info(f'Renamed column {old_name} to {new_name} in table {self.name}, new version: {self.version}')

    def set_comment(self, new_comment: Optional[str]):
        _logger.info(f'[{self.name}] Updating comment: {new_comment}')
        self.comment = new_comment
        self._create_schema_version()

    def set_num_retained_versions(self, new_num_retained_versions: int):
        _logger.info(f'[{self.name}] Updating num_retained_versions: {new_num_retained_versions} (was {self.num_retained_versions})')
        self.num_retained_versions = new_num_retained_versions
        self._create_schema_version()

    def _create_schema_version(self):
        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version
        with Env.get().engine.begin() as conn:
            self._update_md(time.time(), preceding_schema_version, conn)
        _logger.info(f'[{self.name}] Updating table schema to version: {self.version}')

    def insert(
            self, rows: List[Dict[str, Any]], print_stats: bool = False, fail_on_exception : bool = True
    ) -> UpdateStatus:
        """Insert rows into this table.
        """
        assert self.is_insertable()
        from pixeltable.plan import Planner
        plan = Planner.create_insert_plan(self, rows, ignore_errors=not fail_on_exception)
        with Env.get().engine.begin() as conn:
            return self._insert(plan, conn, time.time(), print_stats)

    def _insert(
            self, exec_plan: exec.ExecNode, conn: sql.engine.Connection, timestamp: float, print_stats: bool = False,
    ) -> UpdateStatus:
        """Insert rows produced by exec_plan and propagate to views"""
        # we're creating a new version
        self.version += 1
        result = UpdateStatus()
        num_rows, num_excs, cols_with_excs = self.store_tbl.insert_rows(exec_plan, conn, v_min=self.version)
        self.next_rowid = num_rows
        result.num_rows = num_rows
        result.num_excs = num_excs
        result.num_computed_values += exec_plan.ctx.num_computed_exprs * num_rows
        result.cols_with_excs = [f'{self.name}.{self.cols_by_id[cid].name}' for cid in cols_with_excs]
        self._update_md(timestamp, None, conn)

        # update views
        for view in self.mutable_views:
            from pixeltable.plan import Planner
            plan, _ = Planner.create_view_load_plan(view.path, propagates_insert=True)
            status = view._insert(plan, conn, timestamp, print_stats)
            result.num_rows += status.num_rows
            result.num_excs += status.num_excs
            result.num_computed_values += status.num_computed_values
            result.cols_with_excs += status.cols_with_excs

        result.cols_with_excs = list(dict.fromkeys(result.cols_with_excs).keys())  # remove duplicates
        if print_stats:
            plan.ctx.profile.print(num_rows=num_rows)
        _logger.info(f'TableVersion {self.name}: new version {self.version}')
        return result

    def update(
            self, update_targets: dict[Column, 'pixeltable.exprs.Expr'],
            where_clause: Optional['pixeltable.exprs.Predicate'] = None, cascade: bool = True
    ) -> UpdateStatus:
        with Env.get().engine.begin() as conn:
            return self._update(conn, update_targets, where_clause, cascade)

    def batch_update(
            self, batch: list[dict[Column, 'pixeltable.exprs.Expr']], rowids: list[Tuple[int, ...]],
            cascade: bool = True
    ) -> UpdateStatus:
        """Update rows in batch.
        Args:
            batch: one dict per row, each mapping Columns to LiteralExprs representing the new values
            rowids: if not empty, one tuple per row, each containing the rowid values for the corresponding row in batch
        """
        # if we do lookups of rowids, we must have one for each row in the batch
        assert len(rowids) == 0 or len(rowids) == len(batch)
        import pixeltable.exprs as exprs
        result_status = UpdateStatus()
        cols_with_excs: set[str] = set()
        updated_cols: set[str] = set()
        pk_cols = self.primary_key_columns()
        use_rowids = len(rowids) > 0

        with Env.get().engine.begin() as conn:
            for i, row in enumerate(batch):
                where_clause: Optional[exprs.Expr] = None
                if use_rowids:
                    # construct Where clause to match rowid
                    num_rowid_cols = len(self.store_tbl.rowid_columns())
                    for col_idx in range(num_rowid_cols):
                        assert len(rowids[i]) == num_rowid_cols, f'len({rowids[i]}) != {num_rowid_cols}'
                        clause = exprs.RowidRef(self, col_idx) == rowids[i][col_idx]
                        if where_clause is None:
                            where_clause = clause
                        else:
                            where_clause = where_clause & clause
                else:
                    # construct Where clause for primary key columns
                    for col in pk_cols:
                        assert col in row
                        clause = exprs.ColumnRef(col) == row[col]
                        if where_clause is None:
                            where_clause = clause
                        else:
                            where_clause = where_clause & clause

                update_targets = {col: row[col] for col in row if col not in pk_cols}
                status = self._update(conn, update_targets, where_clause, cascade, show_progress=False)
                result_status.num_rows += status.num_rows
                result_status.num_excs += status.num_excs
                result_status.num_computed_values += status.num_computed_values
                cols_with_excs.update(status.cols_with_excs)
                updated_cols.update(status.updated_cols)

            result_status.cols_with_excs = list(cols_with_excs)
            result_status.updated_cols = list(updated_cols)
            return result_status

    def _update(
            self, conn: sql.engine.Connection, update_targets: dict[Column, 'pixeltable.exprs.Expr'],
            where_clause: Optional['pixeltable.exprs.Predicate'] = None, cascade: bool = True,
            show_progress: bool = True
    ) -> UpdateStatus:
        """Update rows in this table.
        Args:
            update_targets: a list of (column, value) pairs specifying the columns to update and their new values.
            where_clause: a Predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns,
                including within views.
        """
        assert not self.is_snapshot
        from pixeltable.plan import Planner
        plan, updated_cols, recomputed_cols = \
            Planner.create_update_plan(self.path, update_targets, [], where_clause, cascade)
        result = self._propagate_update(
            plan, where_clause.sql_expr() if where_clause is not None else None, recomputed_cols,
            base_versions=[], conn=conn, timestamp=time.time(), cascade=cascade, show_progress=show_progress)
        result.updated_cols = updated_cols
        return result

    def _propagate_update(
            self, plan: Optional[exec.ExecNode], where_clause: Optional[sql.ClauseElement],
            recomputed_view_cols: List[Column], base_versions: List[Optional[int]], conn: sql.engine.Connection,
            timestamp: float, cascade: bool, show_progress: bool = True
    ) -> UpdateStatus:
        result = UpdateStatus()
        if plan is not None:
            # we're creating a new version
            self.version += 1
            result.num_rows, result.num_excs, cols_with_excs = \
                self.store_tbl.insert_rows(plan, conn, v_min=self.version, show_progress=show_progress)
            result.cols_with_excs = [f'{self.name}.{self.cols_by_id[cid].name}' for cid in cols_with_excs]
            self.store_tbl.delete_rows(
                self.version, base_versions=base_versions, match_on_vmin=True, where_clause=where_clause, conn=conn)
            self._update_md(timestamp, None, conn)

        if cascade:
            base_versions = [None if plan is None else self.version] + base_versions  # don't update in place
            # propagate to views
            for view in self.mutable_views:
                recomputed_cols = [col for col in recomputed_view_cols if col.tbl is view]
                plan: Optional[exec.ExecNode] = None
                if len(recomputed_cols) > 0:
                    from pixeltable.plan import Planner
                    plan = Planner.create_view_update_plan(view.path, recompute_targets=recomputed_cols)
                status = view._propagate_update(
                    plan, None, recomputed_view_cols, base_versions=base_versions, conn=conn, timestamp=timestamp, cascade=True)
                result.num_rows += status.num_rows
                result.num_excs += status.num_excs
                result.cols_with_excs += status.cols_with_excs

        result.cols_with_excs = list(dict.fromkeys(result.cols_with_excs).keys())  # remove duplicates
        return result

    def delete(self, where: Optional['pixeltable.exprs.Predicate'] = None) -> UpdateStatus:
        """Delete rows in this table.
        Args:
            where: a Predicate to filter rows to delete.
        """
        assert self.is_insertable()
        from pixeltable.plan import Planner
        analysis_info = Planner.analyze(self, where)
        with Env.get().engine.begin() as conn:
            num_rows = self._delete(analysis_info.sql_where_clause, base_versions=[], conn=conn, timestamp=time.time())

        status = UpdateStatus(num_rows=num_rows)
        return status

    def _delete(
            self, where: Optional['pixeltable.exprs.Predicate'], base_versions: List[Optional[int]],
            conn: sql.engine.Connection, timestamp: float) -> int:
        """Delete rows in this table and propagate to views.
        Args:
            where: a Predicate to filter rows to delete.
        Returns:
            number of deleted rows
        """
        sql_where_clause = where.sql_expr() if where is not None else None
        num_rows = self.store_tbl.delete_rows(
            self.version + 1, base_versions=base_versions, match_on_vmin=False, where_clause=sql_where_clause,
            conn=conn)
        if num_rows > 0:
            # we're creating a new version
            self.version += 1
            self._update_md(timestamp, None, conn)
        else:
            pass
        for view in self.mutable_views:
            num_rows += view._delete(
                where=None, base_versions=[self.version] + base_versions, conn=conn, timestamp=timestamp)
        return num_rows

    def revert(self) -> None:
        """Reverts the table to the previous version.
        """
        assert not self.is_snapshot
        if self.version == 0:
            raise excs.Error('Cannot revert version 0')
        with orm.Session(Env.get().engine, future=True) as session:
            self._revert(session)
            session.commit()

    def _delete_column(self, col: Column, conn: sql.engine.Connection) -> None:
        """Physically remove the column from the schema and the store table"""
        if col.is_stored:
            self.store_tbl.drop_column(col, conn)
        self.cols.remove(col)
        if col.name is not None:
            del self.cols_by_name[col.name]
        del self.cols_by_id[col.id]

    def _revert(self, session: orm.Session) -> None:
        """Reverts this table version and propagates to views"""
        conn = session.connection()
        # make sure we don't have a snapshot referencing this version
        # (unclear how to express this with sqlalchemy)
        query = (
            f"select ts.dir_id, ts.md->'name' "
            f"from {schema.Table.__tablename__} ts "
            f"cross join lateral jsonb_path_query(md, '$.view_md.base_versions[*]') as tbl_version "
            f"where tbl_version->>0 = '{self.id.hex}' and (tbl_version->>1)::int = {self.version}"
        )
        result = list(conn.execute(sql.text(query)))
        if len(result) > 0:
            names = [row[1] for row in result]
            raise excs.Error((
                f'Current version is needed for {len(result)} snapshot{"s" if len(result) > 1 else ""} '
                f'({", ".join(names)})'
            ))

        conn = session.connection()
        # delete newly-added data
        MediaStore.delete(self.id, version=self.version)
        conn.execute(sql.delete(self.store_tbl.sa_tbl).where(self.store_tbl.sa_tbl.c.v_min == self.version))

        # revert new deletions
        set_clause = {self.store_tbl.sa_tbl.c.v_max: schema.Table.MAX_VERSION}
        for index_info in self.idxs_by_name.values():
            # copy the index value back from the undo column and reset the undo column to NULL
            set_clause[index_info.val_col.sa_col] = index_info.undo_col.sa_col
            set_clause[index_info.undo_col.sa_col] = None
        stmt = sql.update(self.store_tbl.sa_tbl) \
            .values(set_clause) \
            .where(self.store_tbl.sa_tbl.c.v_max == self.version)
        conn.execute(stmt)

        # revert schema changes
        if self.version == self.schema_version:
            # delete newly-added columns
            added_cols = [col for col in self.cols if col.schema_version_add == self.schema_version]
            if len(added_cols) > 0:
                next_col_id = min(col.id for col in added_cols)
                for col in added_cols:
                    self._delete_column(col, conn)
                self.next_col_id = next_col_id

            # remove newly-added indices from the lookup structures
            # (the value and undo columns got removed in the preceding step)
            added_idx_md = [md for md in self.idx_md.values() if md.schema_version_add == self.schema_version]
            if len(added_idx_md) > 0:
                next_idx_id = min(md.id for md in added_idx_md)
                for md in added_idx_md:
                    del self.idx_md[md.id]
                    del self.idxs_by_name[md.name]
                self.next_idx_id = next_idx_id

            # make newly-dropped columns visible again
            dropped_cols = [col for col in self.cols if col.schema_version_drop == self.schema_version]
            for col in dropped_cols:
                col.schema_version_drop = None

            # make newly-dropped indices visible again
            dropped_idx_md = [md for md in self.idx_md.values() if md.schema_version_drop == self.schema_version]
            for md in dropped_idx_md:
                md.schema_version_drop = None

            # we need to determine the preceding schema version and reload the schema
            schema_version_md_dict = session.query(schema.TableSchemaVersion.md) \
                .where(schema.TableSchemaVersion.tbl_id == self.id) \
                .where(schema.TableSchemaVersion.schema_version == self.schema_version) \
                .scalar()
            preceding_schema_version = schema_version_md_dict['preceding_schema_version']
            preceding_schema_version_md_dict = session.query(schema.TableSchemaVersion.md) \
                .where(schema.TableSchemaVersion.tbl_id == self.id) \
                .where(schema.TableSchemaVersion.schema_version == preceding_schema_version) \
                .scalar()
            preceding_schema_version_md = schema.md_from_dict(
                schema.TableSchemaVersionMd, preceding_schema_version_md_dict)
            tbl_md = self._create_tbl_md()
            self._init_schema(tbl_md, preceding_schema_version_md)

            conn.execute(
                sql.delete(schema.TableSchemaVersion.__table__)
                    .where(schema.TableSchemaVersion.tbl_id == self.id)
                    .where(schema.TableSchemaVersion.schema_version == self.schema_version))
            self.schema_version = preceding_schema_version
            self.comment = preceding_schema_version_md.comment
            self.num_retained_versions = preceding_schema_version_md.num_retained_versions

        conn.execute(
            sql.delete(schema.TableVersion.__table__)
                .where(schema.TableVersion.tbl_id == self.id)
                .where(schema.TableVersion.version == self.version)
        )
        self.version -= 1
        conn.execute(
            sql.update(schema.Table.__table__)
                .values({schema.Table.md: dataclasses.asdict(self._create_tbl_md())})
                .where(schema.Table.id == self.id))

        # propagate to views
        for view in self.mutable_views:
            view._revert(session)
        _logger.info(f'TableVersion {self.name}: reverted to version {self.version}')

    @classmethod
    def _init_remote(cls, remote_md: dict[str, Any]) -> Tuple[pixeltable.datatransfer.Remote, dict[str, str]]:
        remote_cls = resolve_symbol(remote_md['class'])
        assert isinstance(remote_cls, type) and issubclass(remote_cls, pixeltable.datatransfer.Remote)
        remote = remote_cls.from_dict(remote_md['remote_md'])
        col_mapping = remote_md['col_mapping']
        return remote, col_mapping

    def link(self, remote: pixeltable.datatransfer.Remote, col_mapping: dict[str, str]) -> None:
        # All of the media columns being linked need to either be stored, computed columns or have stored proxies.
        # This ensures that the media in those columns resides in the media cache, where it can be served.
        # First determine which columns (if any) need stored proxies, but don't have one yet.
        cols_by_name = self.path.cols_by_name()  # Includes base columns
        stored_proxies_needed = []
        for col_name in col_mapping.keys():
            col = cols_by_name[col_name]
            if col.col_type.is_media_type() and not (col.is_stored and col.compute_func) and not col.stored_proxy:
                stored_proxies_needed.append(col)
        with Env.get().engine.begin() as conn:
            self.version += 1
            self.remotes[remote] = col_mapping
            preceding_schema_version = None
            if len(stored_proxies_needed) > 0:
                _logger.info(f'Creating stored proxies for columns: {[col.name for col in stored_proxies_needed]}')
                # Create stored proxies for columns that need one. Increment the schema version
                # accordingly.
                preceding_schema_version = self.schema_version
                self.schema_version = self.version
                proxy_cols = [self.create_stored_proxy(col) for col in stored_proxies_needed]
                # Add the columns; this will also update table metadata.
                # TODO Add to base tables
                self._add_columns(proxy_cols, conn)
                # We don't need to retain `UpdateStatus` since the stored proxies are intended to be
                # invisible to the user.
            self._update_md(time.time(), preceding_schema_version, conn)

    def create_stored_proxy(self, col: Column) -> Column:
        from pixeltable import exprs

        assert col.col_type.is_media_type() and not (col.is_stored and col.compute_func) and not col.stored_proxy
        proxy_col = Column(
            name=None,
            computed_with=exprs.ColumnRef(col).apply(lambda x: x, col_type=col.col_type),
            stored=True,
            col_id=self.next_col_id,
            sa_col_type=col.col_type.to_sa_type(),
            schema_version_add=self.schema_version
        )
        proxy_col.tbl = self
        self.next_col_id += 1
        col.stored_proxy = proxy_col
        proxy_col.proxy_base = col
        return proxy_col

    def unlink(self, remote: pixeltable.datatransfer.Remote) -> None:
        assert remote in self.remotes
        timestamp = time.time()
        this_remote_col_names = list(self.remotes[remote].keys())
        other_remote_col_names = {
            col_name
            for other_remote, col_mapping in self.remotes.items() if other_remote != remote
            for col_name in col_mapping.keys()
        }
        cols_by_name = self.path.cols_by_name()  # Includes base columns
        stored_proxy_deletions_needed = [
            cols_by_name[col_name]
            for col_name in this_remote_col_names
            if col_name not in other_remote_col_names and cols_by_name[col_name].stored_proxy
        ]
        with Env.get().engine.begin() as conn:
            self.version += 1
            del self.remotes[remote]
            preceding_schema_version = None
            if len(stored_proxy_deletions_needed) > 0:
                preceding_schema_version = self.schema_version
                self.schema_version = self.version
                proxy_cols = [col.stored_proxy for col in stored_proxy_deletions_needed]
                for col in stored_proxy_deletions_needed:
                    assert col.stored_proxy is not None and col.stored_proxy.proxy_base == col
                    col.stored_proxy.proxy_base = None
                    col.stored_proxy = None
                # TODO Drop from base tables
                self._drop_columns(proxy_cols)
            self._update_md(timestamp, preceding_schema_version, conn)

    def get_remotes(self) -> dict[pixeltable.datatransfer.Remote, dict[str, str]]:
        return self.remotes

    def is_view(self) -> bool:
        return self.base is not None

    def is_component_view(self) -> bool:
        return self.iterator_cls is not None

    def is_insertable(self) -> bool:
        """Returns True if this corresponds to an InsertableTable"""
        return not self.is_snapshot and not self.is_view()

    def is_iterator_column(self, col: Column) -> bool:
        """Returns True if col is produced by an iterator"""
        # the iterator columns directly follow the pos column
        return self.is_component_view() and col.id > 0 and col.id < self.num_iterator_cols + 1

    def is_system_column(self, col: Column) -> bool:
        """Return True if column was created by Pixeltable"""
        if col.name == POS_COLUMN_NAME and self.is_component_view():
            return True
        return False

    def user_columns(self) -> List[Column]:
        """Return all non-system columns"""
        return [c for c in self.cols if not self.is_system_column(c)]

    def primary_key_columns(self) -> List[Column]:
        """Return all non-system columns"""
        return [c for c in self.cols if c.is_pk]

    def get_required_col_names(self) -> List[str]:
        """Return the names of all columns for which values must be specified in insert()"""
        assert not self.is_view()
        names = [c.name for c in self.cols_by_name.values() if not c.is_computed and not c.col_type.nullable]
        return names

    def get_computed_col_names(self) -> List[str]:
        """Return the names of all computed columns"""
        names = [c.name for c in self.cols_by_name.values() if c.is_computed]
        return names

    @classmethod
    def _create_value_expr(cls, col: Column, path: 'pixeltable.catalog.TableVersionPath') -> None:
        """
        Create col.value_expr, given col.compute_func.
        Interprets compute_func's parameters to be references to columns and construct ColumnRefs as args.
        Does not update Column.dependent_cols.
        """
        assert col.value_expr is None
        assert col.compute_func is not None
        from pixeltable import exprs
        params = inspect.signature(col.compute_func).parameters
        args: List[exprs.ColumnRef] = []
        for param_name in params:
            param = path.get_column(param_name)
            if param is None:
                raise excs.Error(
                    f'Column {col.name}: Callable parameter refers to an unknown column: {param_name}')
            args.append(exprs.ColumnRef(param))
        fn = func.make_function(
            col.compute_func, return_type=col.col_type, param_types=[arg.col_type for arg in args])
        col.value_expr = fn(*args)

    def _record_value_expr(self, col: Column) -> None:
        """Update Column.dependent_cols for all cols referenced in col.value_expr.
        """
        assert col.value_expr is not None
        from pixeltable.exprs import ColumnRef
        refd_cols = [e.col for e in col.value_expr.subexprs(expr_class=ColumnRef)]
        for refd_col in refd_cols:
            refd_col.dependent_cols.add(col)

    def get_idx_val_columns(self, cols: Iterable[Column]) -> set[Column]:
        result = {info.val_col for col in cols for info in col.get_idx_info().values()}
        return result

    def get_dependent_columns(self, cols: list[Column]) -> set[Column]:
        """
        Return the set of columns that transitively depend on any of the given ones.
        """
        if len(cols) == 0:
            return set()
        result: set[Column] = set()
        for col in cols:
            result.update(col.dependent_cols)
        result.update(self.get_dependent_columns(result))
        return result

    def num_rowid_columns(self) -> int:
        """Return the number of columns of the rowids, without accessing store_tbl"""
        if self.is_component_view():
            return 1 + self.base.num_rowid_columns()
        return 1

    @classmethod
    def _create_column_md(cls, cols: List[Column]) -> dict[int, schema.ColumnMd]:
        column_md: Dict[int, schema.ColumnMd] = {}
        for col in cols:
            value_expr_dict = col.value_expr.as_dict() if col.value_expr is not None else None
            column_md[col.id] = schema.ColumnMd(
                id=col.id, col_type=col.col_type.as_dict(), is_pk=col.is_pk,
                schema_version_add=col.schema_version_add, schema_version_drop=col.schema_version_drop,
                value_expr=value_expr_dict, stored=col.stored,
                proxy_base=col.proxy_base.id if col.proxy_base else None)
        return column_md

    @classmethod
    def _create_remotes_md(cls, remotes: dict['pixeltable.datatransfer.Remote', dict[str, str]]) -> list[dict[str, Any]]:
        return [
            {
                'class': f'{type(remote).__module__}.{type(remote).__qualname__}',
                'remote_md': remote.to_dict(),
                'col_mapping': col_mapping
            }
            for remote, col_mapping in remotes.items()
        ]

    def _create_tbl_md(self) -> schema.TableMd:
        return schema.TableMd(
            name=self.name, current_version=self.version, current_schema_version=self.schema_version,
            next_col_id=self.next_col_id, next_idx_id=self.next_idx_id, next_row_id=self.next_rowid,
            column_md=self._create_column_md(self.cols), index_md=self.idx_md,
            remotes=self._create_remotes_md(self.remotes), view_md=self.view_md)

    def _create_version_md(self, timestamp: float) -> schema.TableVersionMd:
        return schema.TableVersionMd(created_at=timestamp, version=self.version, schema_version=self.schema_version)

    def _create_schema_version_md(self, preceding_schema_version: int) -> schema.TableSchemaVersionMd:
        column_md: Dict[int, schema.SchemaColumn] = {}
        for pos, col in enumerate(self.cols_by_name.values()):
            column_md[col.id] = schema.SchemaColumn(pos=pos, name=col.name)
        # preceding_schema_version to be set by the caller
        return schema.TableSchemaVersionMd(
            schema_version=self.schema_version, preceding_schema_version=preceding_schema_version,
            columns=column_md, num_retained_versions=self.num_retained_versions, comment=self.comment)
