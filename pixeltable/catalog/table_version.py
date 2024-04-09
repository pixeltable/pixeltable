from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
import time
from typing import Optional, List, Dict, Any, Tuple, Type, Set
from uuid import UUID

import sqlalchemy as sql
import sqlalchemy.orm as orm

import pixeltable
import pixeltable.func as func
from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator
from pixeltable.metadata import schema
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.media_store import MediaStore
from .column import Column
from .globals import UpdateStatus, POS_COLUMN_NAME, is_valid_identifier

_logger = logging.getLogger('pixeltable')

class TableVersion:
    """
    TableVersion represents a particular version of a table/view along with its store table:
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
            self.next_rowid = -1
        else:
            assert tbl_md.current_version == self.version
            self.next_col_id = tbl_md.next_col_id
            self.next_rowid = tbl_md.next_row_id
        self.column_history = tbl_md.column_history

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

        # do this after we determined whether we're a component view, and before we create the store table
        self._init_schema(schema_version_md)

    def __hash__(self) -> int:
        return hash(self.id)

    def create_snapshot_copy(self) -> TableVersion:
        """Create a snapshot copy of this TableVersion"""
        assert not self.is_snapshot
        return TableVersion(
            self.id, self._create_md(), self.version,
            self._create_schema_version_md(preceding_schema_version=0),  # preceding_schema_version: dummy value
            is_snapshot=True, base=self.base)

    @classmethod
    def create(
            cls, session: orm.Session, dir_id: UUID, name: str, cols: List[Column], num_retained_versions: int, comment: str,
            base_path: Optional['pixeltable.catalog.TableVersionPath'] = None, view_md: Optional[schema.ViewMd] = None
    ) -> Tuple[UUID, Optional[TableVersion]]:
        # assign ids
        cols_by_name: Dict[str, Column] = {}
        for pos, col in enumerate(cols):
            col.id = pos
            cols_by_name[col.name] = col
            if col.value_expr is None and col.compute_func is not None:
                cls._create_value_expr(col, base_path)
            if col.is_computed:
                col.check_value_expr()

        ts = time.time()
        # create schema.Table
        column_history = {
            col.id: schema.ColumnHistory(col_id=col.id, schema_version_add=0, schema_version_drop=None)
            for col in cols
        }
        table_md = schema.TableMd(
            name=name, current_version=0, current_schema_version=0,
            next_col_id=len(cols), next_row_id=0, column_history=column_history,
            view_md=view_md)
        tbl_record = schema.Table(dir_id=dir_id, md=dataclasses.asdict(table_md))
        session.add(tbl_record)
        session.flush()  # sets tbl_record.id
        assert tbl_record.id is not None

        # create schema.TableVersion
        table_version_md = schema.TableVersionMd(created_at=ts, version=0, schema_version=0)
        tbl_version_record = schema.TableVersion(
            tbl_id=tbl_record.id, version=0, md=dataclasses.asdict(table_version_md))
        session.add(tbl_version_record)

        # create schema.TableSchemaVersion
        column_md: Dict[int, schema.SchemaColumn] = {}
        for pos, col in enumerate(cols):
            # Column.dependent_cols for existing cols is wrong at this point, but init() will set it correctly
            value_expr_dict = col.value_expr.as_dict() if col.value_expr is not None else None
            column_md[col.id] = schema.SchemaColumn(
                pos=pos, name=col.name, col_type=col.col_type.as_dict(),
                is_pk=col.primary_key, value_expr=value_expr_dict, stored=col.stored, is_indexed=col.is_indexed)

        schema_version_md = schema.TableSchemaVersionMd(
            schema_version=0, preceding_schema_version=None, columns=column_md,
            num_retained_versions=num_retained_versions, comment=comment)
        schema_version_record = schema.TableSchemaVersion(
            tbl_id=tbl_record.id, schema_version=0, md=dataclasses.asdict(schema_version_md))
        session.add(schema_version_record)

        # if this is purely a snapshot (it doesn't require any additional storage for columns and it # doesn't have a
        # predicate to apply at runtime), we don't create a physical table and simply use the base's table version path
        if view_md is not None and view_md.is_snapshot and view_md.predicate is None and len(cols) == 0:
            return tbl_record.id, None

        assert (base_path is not None) == (view_md is not None)
        base = base_path.tbl_version if base_path is not None and view_md.is_snapshot else None
        base_path = base_path if base_path is not None and not view_md.is_snapshot else None
        tbl_version = cls(tbl_record.id, table_md, 0, schema_version_md, base=base, base_path=base_path)
        tbl_version.store_tbl.create(session.connection())
        # TODO: create pgvector indices
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

    def _init_schema(self, schema_version_md: schema.TableSchemaVersionMd) -> None:
        """Initialize self.cols as well as self.store_tbl"""
        self.cols = [Column.from_md(col_id, col_md, self) for col_id, col_md in schema_version_md.columns.items()]
        self.cols_by_name = {col.name: col for col in self.cols}
        self.cols_by_id = {col.id: col for col in self.cols}

        # make sure to traverse columns ordered by position = order in which cols were created;
        # this guarantees that references always point backwards
        from pixeltable import exprs
        for col, col_md in zip(self.cols, schema_version_md.columns.values()):
            col.tbl = self
            if col_md.value_expr is not None:
                col.value_expr = exprs.Expr.from_dict(col_md.value_expr)
                self._record_value_expr(col)

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
            self, ts: float, preceding_schema_version: Optional[int], conn: sql.engine.Connection) -> None:
        """Update all recorded metadata in response to a data or schema change.
        Args:
            ts: timestamp of the change
            preceding_schema_version: last schema version if schema change, else None
        """
        conn.execute(
            sql.update(schema.Table.__table__)
                .values({schema.Table.md: dataclasses.asdict(self._create_md())})
                .where(schema.Table.id == self.id))
        version_md = self._create_version_md(ts)
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
        if col.value_expr is not None:
            col.check_value_expr()
            self._record_value_expr(col)

        row_count = self.store_tbl.count()
        if row_count > 0 and not col.col_type.nullable and not col.is_computed:
            raise excs.Error(f'Cannot add non-nullable column "{col.name}" to table {self.name} with existing rows')

        # we're creating a new schema version
        ts = time.time()
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        self.cols.append(col)
        self.cols_by_name[col.name] = col
        self.cols_by_id[col.id] = col
        self.column_history[col.id] = schema.ColumnHistory(col.id, self.schema_version, None)

        with Env.get().engine.begin() as conn:
            self._update_md(ts, preceding_schema_version, conn)
            _logger.info(f'Added column {col.name} to table {self.name}, new version: {self.version}')
            if col.is_stored:
                self.store_tbl.add_column(col, conn)

        print(f'Added column `{col.name}` to table `{self.name}`.')
        if row_count == 0:
            return UpdateStatus()
        if (not col.is_computed or not col.is_stored) and not col.is_indexed:
            return UpdateStatus(num_rows=row_count)
        # compute values for the existing rows and compute embeddings, if this column is indexed;
        # for some reason, it's not possible to run the following updates in the same transaction as the one
        # that we just used to create the metadata (sqlalchemy hangs when exec() tries to run the query)
        from pixeltable.plan import Planner
        plan, value_expr_slot_idx, embedding_slot_idx = Planner.create_add_column_plan(self.path, col)
        plan.ctx.num_rows = row_count
        # TODO: create pgvector index, if col is indexed

        try:
            # TODO: do this in the same transaction as the metadata update
            with Env.get().engine.begin() as conn:
                plan.ctx.conn = conn
                plan.open()
                num_excs = self.store_tbl.load_column(col, plan, value_expr_slot_idx, embedding_slot_idx, conn)
        except sql.exc.DBAPIError as e:
            self.drop_column(col.name)
            raise excs.Error(f'Error during SQL execution:\n{e}')
        finally:
            plan.close()

        msg = f'Added {row_count} column value{"" if row_count == 1 else "s"} with {num_excs} error{"" if num_excs == 1 else "s"}.'
        print(msg)
        _logger.info(f'Column {col.name}: {msg}')
        if print_stats:
            plan.ctx.profile.print(num_rows=row_count)
        return UpdateStatus(
            num_rows=row_count, num_computed_values=row_count, num_excs=num_excs,
            cols_with_excs=[f'{self.name}.{col.name}'] if num_excs > 0 else [])

    def drop_column(self, name: str) -> None:
        """Drop a column from the table.
        """
        assert not self.is_snapshot
        if name not in self.cols_by_name:
            raise excs.Error(f'Unknown column: {name}')
        col = self.cols_by_name[name]
        if len(col.dependent_cols) > 0:
            raise excs.Error(
                f'Cannot drop column {name} because the following columns depend on it:\n',
                f'{", ".join([c.name for c in col.dependent_cols])}')

        if col.value_expr is not None:
            # update Column.dependent_cols
            for c in self.cols:
                if c == col:
                    break
                c.dependent_cols.discard(col)

        # we're creating a new schema version
        ts = time.time()
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        self.cols.remove(col)
        del self.cols_by_name[name]
        del self.cols_by_id[col.id]
        self.column_history[col.id].schema_version_drop = self.schema_version

        with Env.get().engine.begin() as conn:
            self._update_md(ts, preceding_schema_version, conn)
        if col.is_stored:
            self.store_tbl.drop_column()
        _logger.info(f'Dropped column {name} from table {self.name}, new version: {self.version}')

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
        ts = time.time()
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        with Env.get().engine.begin() as conn:
            self._update_md(ts, preceding_schema_version, conn)
        _logger.info(f'Renamed column {old_name} to {new_name} in table {self.name}, new version: {self.version}')

    def set_comment(self, new_comment: Optional[str]):
        _logger.info(f'[{self.name}] Updating comment: {new_comment}')
        self.comment = new_comment
        self._commit_new_schema_version()

    def set_num_retained_versions(self, new_num_retained_versions: int):
        _logger.info(f'[{self.name}] Updating num_retained_versions: {new_num_retained_versions} (was {self.num_retained_versions})')
        self.num_retained_versions = new_num_retained_versions
        self._commit_new_schema_version()

    def _commit_new_schema_version(self):
        # we're creating a new schema version
        ts = time.time()
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version
        with Env.get().engine.begin() as conn:
            self._update_md(ts, preceding_schema_version, conn)
        _logger.info(f'[{self.name}] Updating table schema to version: {self.version}')

    def insert(
            self, rows: List[Dict[str, Any]], print_stats: bool = False, fail_on_exception : bool = True
    ) -> UpdateStatus:
        """Insert rows into this table.
        """
        assert self.is_insertable()
        from pixeltable.plan import Planner
        plan = Planner.create_insert_plan(self, rows, ignore_errors=not fail_on_exception)
        ts = time.time()
        with Env.get().engine.begin() as conn:
            return self._insert(plan, conn, ts, print_stats)

    def _insert(
            self, exec_plan: exec.ExecNode, conn: sql.engine.Connection, ts: float, print_stats: bool = False,
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
        self._update_md(ts, None, conn)

        # update views
        for view in self.mutable_views:
            from pixeltable.plan import Planner
            plan, _ = Planner.create_view_load_plan(view.path, propagates_insert=True)
            status = view._insert(plan, conn, ts, print_stats)
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
            self, update_targets: Optional[List[Tuple[Column, 'pixeltable.exprs.Expr']]] = None,
            where_clause: Optional['pixeltable.exprs.Predicate'] = None, cascade: bool = True
    ) -> UpdateStatus:
        """Update rows in this table.
        Args:
            update_targets: a list of (column, value) pairs specifying the columns to update and their new values.
            where_clause: a Predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns,
                including within views.
        """
        if update_targets is None:
            update_targets = []
        assert not self.is_snapshot
        from pixeltable.plan import Planner
        plan, updated_cols, recomputed_cols = \
            Planner.create_update_plan(self.path, update_targets, [], where_clause, cascade)
        with Env.get().engine.begin() as conn:
            ts = time.time()
            result = self._update(
                plan, where_clause.sql_expr() if where_clause is not None else None, recomputed_cols,
                base_versions=[], conn=conn, ts=ts, cascade=cascade)
            result.updated_cols = updated_cols
            return result

    def _update(
            self, plan: Optional[exec.ExecNode], where_clause: Optional[sql.ClauseElement],
            recomputed_view_cols: List[Column], base_versions: List[Optional[int]], conn: sql.engine.Connection,
            ts: float, cascade: bool
    ) -> UpdateStatus:
        result = UpdateStatus()
        if plan is not None:
            # we're creating a new version
            self.version += 1
            result.num_rows, result.num_excs, cols_with_excs = \
                self.store_tbl.insert_rows(plan, conn, v_min=self.version)
            result.cols_with_excs = [f'{self.name}.{self.cols_by_id[cid].name}' for cid in cols_with_excs]
            self.store_tbl.delete_rows(
                self.version, base_versions=base_versions, match_on_vmin=True, where_clause=where_clause, conn=conn)
            self._update_md(ts, None, conn)

        if cascade:
            base_versions = [None if plan is None else self.version] + base_versions  # don't update in place
            # propagate to views
            for view in self.mutable_views:
                recomputed_cols = [col for col in recomputed_view_cols if col.tbl is view]
                plan: Optional[exec.ExecNode] = None
                if len(recomputed_cols) > 0:
                    from pixeltable.plan import Planner
                    plan = Planner.create_view_update_plan(view.path, recompute_targets=recomputed_cols)
                status = view._update(
                    plan, None, recomputed_view_cols, base_versions=base_versions, conn=conn, ts=ts, cascade=True)
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
        ts = time.time()
        with Env.get().engine.begin() as conn:
            num_rows = self._delete(analysis_info.sql_where_clause, base_versions=[], conn=conn, ts=ts)

        status = UpdateStatus(num_rows=num_rows)
        return status

    def _delete(
            self, where: Optional['pixeltable.exprs.Predicate'], base_versions: List[Optional[int]],
            conn: sql.engine.Connection, ts: float) -> int:
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
            self._update_md(ts, None, conn)
        else:
            pass
        for view in self.mutable_views:
            num_rows += view._delete(where=None, base_versions=[self.version] + base_versions, conn=conn, ts=ts)
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
        conn.execute(
            sql.update(self.store_tbl.sa_tbl) \
                .values({self.store_tbl.sa_tbl.c.v_max: schema.Table.MAX_VERSION})
                .where(self.store_tbl.sa_tbl.c.v_max == self.version))

        if self.version == self.schema_version:
            # the current version involved a schema change:
            # if the schema change was to add a column, we now need to drop it
            added_col_ids = [
                col_history.col_id for col_history in self.column_history.values()
                if col_history.schema_version_add == self.schema_version
            ]
            assert len(added_col_ids) <= 1
            added_col: Optional[Column] = None
            if len(added_col_ids) == 1:
                added_col_id = added_col_ids[0]
                # drop this newly-added column and its ColumnHistory record
                c = self.cols_by_id[added_col_id]
                if c.is_stored:
                    added_col = c
                del self.column_history[c.id]

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
            self._init_schema(preceding_schema_version_md)

            # physically drop the column, but only after we have re-created the schema
            if added_col is not None:
                self.store_tbl.drop_column(added_col, conn)

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
                .values({schema.Table.md: dataclasses.asdict(self._create_md())})
                .where(schema.Table.id == self.id))

        # propagate to views
        for view in self.mutable_views:
            view._revert(session)
        _logger.info(f'TableVersion {self.name}: reverted to version {self.version}')

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

    def get_required_col_names(self) -> List[str]:
        """Return the names of all columns for which values must be specified in insert()"""
        assert not self.is_view()
        names = [c.name for c in self.cols if not c.is_computed and not c.col_type.nullable]
        return names

    def get_computed_col_names(self) -> List[str]:
        """Return the names of all computed columns"""
        names = [c.name for c in self.cols if c.is_computed]
        return names

    @classmethod
    def _create_value_expr(cls, col: Column, path: 'TableVersionPath') -> None:
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

    def get_dependent_columns(self, cols: List[Column]) -> Set[Column]:
        """
        Return the set of columns that transitively depend on any of the given ones.
        """
        if len(cols) == 0:
            return []
        result: Set[Column] = set()
        for col in cols:
            result.update(col.dependent_cols)
        result.update(self.get_dependent_columns(result))
        return result

    def num_rowid_columns(self) -> int:
        """Return the number of columns of the rowids, without accessing store_tbl"""
        if self.is_component_view():
            return 1 + self.base.num_rowid_columns()
        return 1

    def _create_md(self) -> schema.TableMd:
        return schema.TableMd(
            name=self.name, current_version=self.version, current_schema_version=self.schema_version,
            next_col_id=self.next_col_id, next_row_id=self.next_rowid, column_history=self.column_history,
            view_md=self.view_md)

    def _create_version_md(self, ts: float) -> schema.TableVersionMd:
        return schema.TableVersionMd(created_at=ts, version=self.version, schema_version=self.schema_version)

    def _create_schema_version_md(self, preceding_schema_version: int) -> schema.TableSchemaVersionMd:
        column_md: Dict[int, schema.SchemaColumn] = {}
        for pos, col in enumerate(self.cols):
            value_expr_dict = col.value_expr.as_dict() if col.value_expr is not None else None
            column_md[col.id] = schema.SchemaColumn(
                pos=pos, name=col.name, col_type=col.col_type.as_dict(),
                is_pk=col.primary_key, value_expr=value_expr_dict, stored=col.stored, is_indexed=col.is_indexed)
        # preceding_schema_version to be set by the caller
        return schema.TableSchemaVersionMd(
            schema_version=self.schema_version, preceding_schema_version=preceding_schema_version,
            columns=column_md, num_retained_versions=self.num_retained_versions, comment=self.comment)
