from __future__ import annotations

import copy
import dataclasses
import inspect
import logging
from typing import Optional, List, Dict, Any, Union, Tuple, Type, Set
from uuid import UUID
import time
import importlib

import sqlalchemy as sql
import sqlalchemy.orm as orm

from .globals import UpdateStatus, POS_COLUMN_NAME, is_valid_identifier
from .column import Column
from pixeltable import exceptions as exc
from pixeltable.env import Env
import pixeltable.func as func
from pixeltable.metadata import schema
from pixeltable.utils.media_store import MediaStore
from pixeltable.utils.filecache import FileCache
from pixeltable.iterators import ComponentIterator


_logger = logging.getLogger('pixeltable')

class TableVersion:
    """
    TableVersion contains all metadata needed to execute queries and updates against a particular version of a
    table/view (ie, what is recorded in schema.Table):
    - schema information
    - for views, the full chain of base tables

    If this version is not the current version, updates are disabled.
    """

    def __init__(
            self, id: UUID, base: Optional[TableVersion], tbl_md: schema.TableMd, version: int,
            schema_version_md: schema.TableSchemaVersionMd
    ):
        self.id = id
        self.name = tbl_md.name
        self.base = base
        self.version = version
        self.schema_version = schema_version_md.schema_version
        if tbl_md.current_version == self.version:
            self.next_col_id = tbl_md.next_col_id
            self.next_rowid = tbl_md.next_row_id
        else:
            # disable schema changes and updates
            self.next_col_id = -1
            self.next_rowid = -1
        self.column_history = tbl_md.column_history
        self.parameters = tbl_md.parameters

        # view-specific initialization
        from pixeltable import exprs
        self.predicate = exprs.Expr.from_dict(tbl_md.predicate, self) if tbl_md.predicate is not None else None
        self.views: List[TableVersion] = []  # views that reference us
        if self.base is not None:
            self.base.views.append(self)

        # component view-specific initialization
        self.iterator_cls: Optional[Type[ComponentIterator]] = None
        self.num_iterator_cols = 0
        if tbl_md.iterator_class_fqn is not None:
            module_name, class_name = tbl_md.iterator_class_fqn.rsplit('.', 1)
            module = importlib.import_module(module_name)
            self.iterator_cls = getattr(module, class_name)
            output_schema, _ = self.iterator_cls.output_schema()
            self.num_iterator_cols = len(output_schema)
        self.iterator_args: Optional[exprs.InlineDict] = None
        if tbl_md.iterator_args is not None:
            self.iterator_args = exprs.Expr.from_dict(tbl_md.iterator_args, self)

        # do this after we determined whether we're a component view, and before we create the store table
        self._set_cols(schema_version_md)

        from pixeltable.store import StoreTable, StoreView, StoreComponentView
        if self.is_component_view():
            self.store_tbl = StoreComponentView(self)
        elif self.is_view():
            self.store_tbl = StoreView(self)
        else:
            self.store_tbl = StoreTable(self)

    def __hash__(self) -> int:
        return hash(self.id)

    def create_snapshot_copy(self) -> TableVersion:
        """Create an immutable copy of this TableVersion for a particular snapshot"""
        result = TableVersion(
            self.id, self.base, self._create_md(), self.version,
            self._create_schema_version_md(preceding_schema_version=0))  # preceding_schema_version: dummy value
        result.next_col_id = -1
        result.next_rowid = -1
        return result

    @classmethod
    def create(
            cls, dir_id: UUID, name: str, cols: List[Column],
            base: Optional[TableVersion], predicate: Optional['exprs.Predicate'], num_retained_versions: int,
            iterator_cls: Optional[Type[ComponentIterator]],
            iterator_args: Optional['exprs.InlineDict'],
            session: orm.Session
    ) -> TableVersion:
        # create a copy here so we can modify it
        cols = [copy.copy(c) for c in cols]
        # assign ids
        cols_by_name: Dict[str, Column] = {}
        for pos, col in enumerate(cols):
            col.id = pos
            cols_by_name[col.name] = col
            if col.value_expr is None and col.compute_func is not None:
                cls._create_value_expr(col, cols_by_name)
            if col.is_computed:
                col.check_value_expr()

        params = schema.TableParameters(num_retained_versions)

        ts = time.time()
        # create schema.MutableTable
        column_history = {
            col.id: schema.ColumnHistory(col_id=col.id, schema_version_add=0, schema_version_drop=None)
            for col in cols
        }
        iterator_class_fqn = f'{iterator_cls.__module__}.{iterator_cls.__name__}' if iterator_cls is not None else None
        table_md = schema.TableMd(
            name=name, parameters=params, current_version=0, current_schema_version=0,
            next_col_id=len(cols), next_row_id=0, column_history=column_history,
            predicate=predicate.as_dict() if predicate is not None else None,
            iterator_class_fqn=iterator_class_fqn,
            iterator_args=iterator_args.as_dict() if iterator_args is not None else None,
        )
        # base version: if we're referencing a live table, the base version is None
        base_version = None if base is None or base.next_rowid != -1 else base.version
        tbl_record = schema.Table(
            dir_id=dir_id, base_id=base.id if base is not None else None, base_version=base_version,
            md=dataclasses.asdict(table_md))
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
            # Column.dependent_cols for existing cols is wrong at this point, but MutableTable.init() will set it correctly
            value_expr_dict = col.value_expr.as_dict() if col.value_expr is not None else None
            column_md[col.id] = schema.SchemaColumn(
                pos=pos, name=col.name, col_type=col.col_type.as_dict(),
                is_pk=col.primary_key, value_expr=value_expr_dict, stored=col.stored, is_indexed=col.is_indexed)

        schema_version_md = schema.TableSchemaVersionMd(
            schema_version=0, preceding_schema_version=None, columns=column_md)
        schema_version_record = schema.TableSchemaVersion(
            tbl_id=tbl_record.id, schema_version=0, md=dataclasses.asdict(schema_version_md))
        session.add(schema_version_record)

        tbl_version = cls(tbl_record.id, base, table_md, 0, schema_version_md)
        tbl_version.store_tbl.create(session.connection())
        # TODO: create pgvector indices
        return tbl_version

    def drop(self) -> None:
        with orm.Session(Env.get().engine, future=True) as session:
            # check if we have snapshots
            num_references = session.query(sql.func.count(schema.TableSnapshot.id)) \
                .where(schema.TableSnapshot.tbl_id == self.id) \
                .scalar()
            if num_references > 0:
                raise exc.Error((
                    f'Cannot drop table {self.name}, which has {num_references} snapshot'
                    f'{"s" if num_references > 1 else ""}'
                ))
            # check if we have views
            num_references = session.query(sql.func.count(schema.Table.id)) \
                .where(schema.Table.base_id == self.id) \
                .scalar()
            if num_references > 0:
                raise exc.Error((
                    f'Cannot drop table {self.name}, which has {num_references} views'
                    f'{"s" if num_references > 1 else ""}'
                ))

            # delete this table and all associated data
            MediaStore.delete(self.id)
            FileCache.get().clear(tbl_id=self.id)
            conn = session.connection()
            conn.execute(
                sql.delete(schema.TableSchemaVersion.__table__).where(schema.TableSchemaVersion.tbl_id == self.id))
            conn.execute(
                sql.delete(schema.TableVersion.__table__).where(schema.TableVersion.tbl_id == self.id))
            conn.execute(sql.delete(schema.Table.__table__).where(schema.Table.id == self.id))
            self.store_tbl.drop(conn)
            session.commit()

    def _set_cols(self, schema_version_md: schema.TableSchemaVersionMd) -> None:
        self.cols = [Column.from_md(col_id, col_md, self) for col_id, col_md in schema_version_md.columns.items()]
        self.cols_by_name = {col.name: col for col in self.cols}
        self.cols_by_id = {col.id: col for col in self.cols}

        # make sure to traverse columns ordered by position = order in which cols were created;
        # this guarantees that references always point backwards
        from pixeltable import exprs
        for col, col_md in zip(self.cols, schema_version_md.columns.values()):
            col.tbl = self
            if col_md.value_expr is not None:
                col.value_expr = exprs.Expr.from_dict(col_md.value_expr, self)
                self._record_value_expr(col)

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
        assert self.next_col_id != -1
        assert is_valid_identifier(col.name)
        assert col.stored is not None
        assert col.name not in self.cols_by_name
        col.tbl = self
        col.id = self.next_col_id
        self.next_col_id += 1

        if col.compute_func is not None:
            # create value_expr from compute_func
            self._create_value_expr(col, self.cols_by_name)
        if col.value_expr is not None:
            col.check_value_expr()
            self._record_value_expr(col)

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

        from pixeltable.dataframe import DataFrame
        row_count = DataFrame(self).count()
        if row_count == 0:
            return UpdateStatus()
        if (not col.is_computed or not col.is_stored) and not col.is_indexed:
            return UpdateStatus(num_rows=row_count)
        # compute values for the existing rows and compute embeddings, if this column is indexed;
        # for some reason, it's not possible to run the following updates in the same transaction as the one
        # that we just used to create the metadata (sqlalchemy hangs when exec() tries to run the query)
        from pixeltable.plan import Planner
        plan, value_expr_slot_idx, embedding_slot_idx = Planner.create_add_column_plan(self, col)
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
            raise exc.Error(f'Error during SQL execution:\n{e}')
        finally:
            plan.close()

        msg = f'added {row_count} column values with {num_excs} error{"" if num_excs == 1 else "s"}'
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
        if name not in self.cols_by_name:
            raise exc.Error(f'Unknown column: {name}')
        col = self.cols_by_name[name]
        if len(col.dependent_cols) > 0:
            raise exc.Error(
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
        if old_name not in self.cols_by_name:
            raise exc.Error(f'Unknown column: {old_name}')
        if not is_valid_identifier(new_name):
            raise exc.Error(f"Invalid column name: '{new_name}'")
        if new_name in self.cols_by_name:
            raise exc.Error(f'Column {new_name} already exists')
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
        assert self.is_mutable()
        # we're creating a new version
        self.version += 1
        from pixeltable.plan import Planner
        result = UpdateStatus()
        num_rows, num_excs, cols_with_excs = self.store_tbl.insert_rows(exec_plan, conn, v_min=self.version)
        self.next_rowid = num_rows
        result.num_rows = num_rows
        result.num_excs = num_excs
        result.num_computed_values += exec_plan.ctx.num_computed_exprs * num_rows
        result.cols_with_excs = [f'{self.name}.{self.cols_by_id[cid].name}' for cid in cols_with_excs]
        self._update_md(ts, None, conn)

        # update views
        for view in self.views:
            from pixeltable.plan import Planner
            plan, _ = Planner.create_view_load_plan(view, propagates_insert=True)
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
            self, update_targets: List[Tuple[Column, 'pixeltable.exprs.Expr']] = [],
        where_clause: Optional['pixeltable.exprs.Predicate'] = None, cascade: bool = True
    ) -> UpdateStatus:
        """Update rows in this table.
        Args:
            update_targets: a list of (column, value) pairs specifying the columns to update and their new values.
            where_clause: a Predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns,
                including within views.
        """
        assert self.is_mutable()
        from pixeltable.plan import Planner
        plan, updated_cols, recomputed_cols = \
            Planner.create_update_plan(self, update_targets, [], where_clause, cascade)
        plan.ctx.set_pk_clause(self.store_tbl.pk_columns())
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
            for view in self.views:
                recomputed_cols = [col for col in recomputed_view_cols if col.tbl == view]
                plan: Optional[exec.ExecNode] = None
                if len(recomputed_cols) > 0:
                    from pixeltable.plan import Planner
                    plan = Planner.create_view_update_plan(view, recompute_targets=recomputed_cols)
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
            self.version + 1, base_versions=base_versions, match_on_vmin=False, where_clause=sql_where_clause, conn=conn)
        if num_rows > 0:
            # we're creating a new version
            self.version += 1
            self._update_md(ts, None, conn)
        else:
            pass
        for view in self.views:
            num_rows += view._delete(where=None, base_versions=[self.version] + base_versions, conn=conn, ts=ts)
        return num_rows

    def revert(self) -> None:
        """Reverts the table to the previous version.
        """
        assert self.is_mutable()
        if self.version == 0:
            raise exc.Error('Cannot revert version 0')
        with orm.Session(Env.get().engine, future=True) as session:
            self._revert(session)
            session.commit()

    def _revert(self, session: orm.Session) -> None:
        """Reverts this table version and propagates to views"""
        # make sure we don't have a snapshot referencing this version
        num_references = session.query(sql.func.count(schema.TableSnapshot.id)) \
            .where(schema.TableSnapshot.tbl_id == self.id) \
            .where(schema.TableSnapshot.tbl_version == self.version) \
            .scalar()
        if num_references > 0:
            raise exc.Error(
                f'Current version is needed for {num_references} snapshot{"s" if num_references > 1 else ""}')

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
            self._set_cols(preceding_schema_version_md)

            # physically drop the column, but only after we have re-created the schema
            if added_col is not None:
                self.store_tbl.drop_column(added_col, conn)

            conn.execute(
                sql.delete(schema.TableSchemaVersion.__table__)
                    .where(schema.TableSchemaVersion.tbl_id == self.id)
                    .where(schema.TableSchemaVersion.schema_version == self.schema_version))
            self.schema_version = preceding_schema_version

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
        for view in self.views:
            view._revert(session)
        _logger.info(f'TableVersion {self.name}: reverted to version {self.version}')

    def is_view(self) -> bool:
        return self.base is not None

    def is_component_view(self) -> bool:
        return self.is_view() and self.iterator_cls is not None

    def is_insertable(self) -> bool:
        """Returns True if this corresponds to an InsertableTable"""
        return self.next_rowid != -1 and not self.is_view()

    def is_mutable(self) -> bool:
        """Returns True if this corresponds to a MutableTable"""
        return self.next_rowid != -1

    def get_bases(self) -> List[TableVersion]:
        """Return all bases"""
        if self.base is None:
            return []
        return [self.base] + self.base.get_bases()

    def find_tbl(self, id: UUID) -> Optional[TableVersion]:
        """Return the matching TableVersion in the chain of TableVersions, starting with this one"""
        if self.id == id:
            return self
        if self.base is None:
            return None
        return self.base.find_tbl(id)

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
        assert not self.is_view()
        names = [c.name for c in self.cols if c.is_computed]
        return names

    def check_input_rows(self, rows: List[List[Any]], column_names: List[str]) -> None:
        """
        Verify the integrity of 'rows':
        1. the number of columns matches the number of values provided
        2. all required table columns are present
        3. all columns provided are insertable (ie not computed)
        4. all provided values for a column are of a compatible python type
        """
        assert len(rows) > 0
        all_col_names = {col.name for col in self.cols}
        reqd_col_names = set(self.get_required_col_names(required_only=True))
        given_col_names = set(column_names)
        if not(reqd_col_names <= given_col_names):
            raise exc.Error(f'Missing columns: {", ".join(reqd_col_names - given_col_names)}')
        if not(given_col_names <= all_col_names):
            raise exc.Error(f'Unknown columns: {", ".join(given_col_names - all_col_names)}')
        computed_col_names = {col.name for col in self.cols if col.value_expr is not None}
        if len(computed_col_names & given_col_names) > 0:
            raise exc.Error(
                f'Provided values for computed columns: {", ".join(computed_col_names & given_col_names)}')

        # check data
        row_cols = [self.cols_by_name[name] for name in column_names]
        for col_idx, col in enumerate(row_cols):
            for row_idx, row in enumerate(rows):
                if not col.col_type.nullable and row[col_idx] is None:
                    raise exc.Error(
                        f'Column {col.name}: row {row_idx} contains None for a non-nullable column')
                val = row[col_idx]
                if val is None:
                    continue
                try:
                    # basic sanity checks here
                    checked_val = col.col_type.create_literal(val)
                    row[col_idx] = checked_val
                except TypeError as e:
                    raise exc.Error(f'Column {col.name} in row {row_idx}: {e}')

    @classmethod
    def _create_value_expr(cls, col: Column, existing_cols: Dict[str, Column]) -> None:
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
            if param_name not in existing_cols:
                raise exc.Error(
                    f'Column {col.name}: compute_with parameter refers to an unknown column: {param_name}')
            args.append(exprs.ColumnRef(existing_cols[param_name]))
        fn = func.make_function(col.col_type, [arg.col_type for arg in args], col.compute_func)
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

    def _create_md(self) -> schema.TableMd:
        iterator_class_fqn = \
            f'{self.iterator_cls.__module__}.{self.iterator_cls.__name__}' if self.iterator_cls is not None else None
        return schema.TableMd(
            name=self.name, current_version=self.version, current_schema_version=self.schema_version,
            next_col_id=self.next_col_id, next_row_id=self.next_rowid, column_history=self.column_history,
            parameters=self.parameters, predicate=self.predicate.as_dict() if self.predicate is not None else None,
            iterator_class_fqn=iterator_class_fqn,
            iterator_args = self.iterator_args.as_dict() if self.iterator_args is not None else None,
        )

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
            columns=column_md)

    def __getattr__(self, col_name: str) -> 'pixeltable.exprs.ColumnRef':
        """Return a ColumnRef for the given column name."""
        from pixeltable.exprs import ColumnRef, RowidRef
        if col_name == POS_COLUMN_NAME and self.is_component_view():
            return RowidRef(self, self.store_tbl.pos_col_idx)
        if col_name not in self.cols_by_name:
            if self.base is None:
                raise AttributeError(f'Column {col_name} unknown')
            return getattr(self.base, col_name)
        col = self.cols_by_name[col_name]
        return ColumnRef(col)

    def __getitem__(self, index: object) -> Union['pixeltable.exprs.ColumnRef', 'pixeltable.dataframe.DataFrame']:
        """Return a ColumnRef for the given column name, or a DataFrame for the given slice.
        """
        if isinstance(index, str):
            # basically <tbl>.<colname>
            return self.__getattr__(index)
        from pixeltable.dataframe import DataFrame
        return DataFrame(self).__getitem__(index)

    def columns(self) -> List[Column]:
        """Return all columns visible in this table, including columns from bases"""
        result = self.cols.copy()
        if self.base is not None:
            base_cols = self.base.columns()
            # we only include base columns that don't conflict with one of our column names
            result.extend([c for c in base_cols if c.name not in self.cols_by_name])
        return result

    def get_column(self, name: str) -> Optional[Column]:
        """Return the column with the given name, or None if not found"""
        col = self.cols_by_name.get(name)
        if col is not None:
            return col
        elif self.base is not None:
            return self.base.get_column(name)
        else:
            return None

    def has_column(self, col: Column) -> bool:
        """Return True if this table has the given column.
        """
        assert col.tbl is not None
        if col.tbl == self:
            return True
        elif self.base is not None:
            return self.base.has_column(col)
        else:
            return False

