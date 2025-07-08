from __future__ import annotations

import abc
import logging
import sys
import warnings
from typing import Any, Iterable, Iterator, Optional, Union

import more_itertools
import sqlalchemy as sql
from tqdm import TqdmWarning, tqdm

from pixeltable import catalog, exceptions as excs
from pixeltable.catalog.update_status import RowCountStats
from pixeltable.env import Env
from pixeltable.exec import ExecNode
from pixeltable.metadata import schema
from pixeltable.utils.exception_handler import run_cleanup
from pixeltable.utils.media_store import MediaStore
from pixeltable.utils.sql import log_explain, log_stmt

_logger = logging.getLogger('pixeltable')


class StoreBase:
    """Base class for stored tables

    Each row has the following system columns:
    - rowid columns: one or more columns that identify a user-visible row across all versions
    - v_min: version at which the row was created
    - v_max: version at which the row was deleted (or MAX_VERSION if it's still live)
    """

    tbl_version: catalog.TableVersionHandle
    sa_md: sql.MetaData
    sa_tbl: Optional[sql.Table]
    _pk_cols: list[sql.Column]
    v_min_col: sql.Column
    v_max_col: sql.Column
    base: Optional[StoreBase]

    # In my cursory experiments this was the optimal batch size: it was an improvement over 5_000 and there was no real
    # benefit to going higher.
    # TODO: Perform more rigorous experiments with different table structures and OS environments to refine this.
    __INSERT_BATCH_SIZE = 10_000

    def __init__(self, tbl_version: catalog.TableVersion):
        self.tbl_version = catalog.TableVersionHandle(
            tbl_version.id, tbl_version.effective_version, tbl_version=tbl_version
        )
        self.sa_md = sql.MetaData()
        self.sa_tbl = None
        # We need to declare a `base` variable here, even though it's only defined for instances of `StoreView`,
        # since it's referenced by various methods of `StoreBase`
        self.base = tbl_version.base.get().store_tbl if tbl_version.base is not None else None
        # we're passing in tbl_version to avoid a circular call to TableVersionHandle.get()
        self.create_sa_tbl(tbl_version)

    def system_columns(self) -> list[sql.Column]:
        return [*self._pk_cols, self.v_max_col]

    def pk_columns(self) -> list[sql.Column]:
        return self._pk_cols

    def rowid_columns(self) -> list[sql.Column]:
        return self._pk_cols[:-1]

    @abc.abstractmethod
    def _create_rowid_columns(self) -> list[sql.Column]:
        """Create and return rowid columns"""

    def _create_system_columns(self) -> list[sql.Column]:
        """Create and return system columns"""
        rowid_cols = self._create_rowid_columns()
        self.v_min_col = sql.Column('v_min', sql.BigInteger, nullable=False)
        self.v_max_col = sql.Column(
            'v_max', sql.BigInteger, nullable=False, server_default=str(schema.Table.MAX_VERSION)
        )
        self._pk_cols = [*rowid_cols, self.v_min_col]
        return [*rowid_cols, self.v_min_col, self.v_max_col]

    def create_sa_tbl(self, tbl_version: Optional[catalog.TableVersion] = None) -> None:
        """Create self.sa_tbl from self.tbl_version."""
        if tbl_version is None:
            tbl_version = self.tbl_version.get()
        system_cols = self._create_system_columns()
        all_cols = system_cols.copy()
        for col in [c for c in tbl_version.cols if c.is_stored]:
            # re-create sql.Column for each column, regardless of whether it already has sa_col set: it was bound
            # to the last sql.Table version we created and cannot be reused
            col.create_sa_cols()
            all_cols.append(col.sa_col)
            if col.stores_cellmd:
                all_cols.append(col.sa_cellmd_col)

        if self.sa_tbl is not None:
            # if we're called in response to a schema change, we need to remove the old table first
            self.sa_md.remove(self.sa_tbl)

        idxs: list[sql.Index] = []
        # index for all system columns:
        # - base x view joins can be executed as merge joins
        # - speeds up ORDER BY rowid DESC
        # - allows filtering for a particular table version in index scan
        idx_name = f'sys_cols_idx_{tbl_version.id.hex}'
        idxs.append(sql.Index(idx_name, *system_cols))

        # v_min/v_max indices: speeds up base table scans needed to propagate a base table insert or delete
        idx_name = f'vmin_idx_{tbl_version.id.hex}'
        idxs.append(sql.Index(idx_name, self.v_min_col, postgresql_using=Env.get().dbms.version_index_type))
        idx_name = f'vmax_idx_{tbl_version.id.hex}'
        idxs.append(sql.Index(idx_name, self.v_max_col, postgresql_using=Env.get().dbms.version_index_type))

        self.sa_tbl = sql.Table(self._storage_name(), self.sa_md, *all_cols, *idxs)
        # _logger.debug(f'created sa tbl for {tbl_version.id!s} (sa_tbl={id(self.sa_tbl):x}, tv={id(tbl_version):x})')

    @abc.abstractmethod
    def _rowid_join_predicate(self) -> sql.ColumnElement[bool]:
        """Return predicate for rowid joins to all bases"""

    @abc.abstractmethod
    def _storage_name(self) -> str:
        """Return the name of the data store table"""

    def _move_tmp_media_file(self, file_url: Optional[str], col: catalog.Column, v_min: int) -> str:
        return MediaStore.move_tmp_media_file(file_url, self.tbl_version.id, col.id, v_min)

    def _move_tmp_media_files(
        self, table_row: list[Any], media_cols_by_sql_idx: dict[int, catalog.Column], v_min: int
    ) -> None:
        """Move tmp media files that we generated to a permanent location"""
        for n, col in media_cols_by_sql_idx.items():
            table_row[n] = self._move_tmp_media_file(table_row[n], col, v_min)

    def count(self) -> int:
        """Return the number of rows visible in self.tbl_version"""
        stmt = (
            sql.select(sql.func.count('*'))
            .select_from(self.sa_tbl)
            .where(self.v_min_col <= self.tbl_version.get().version)
            .where(self.v_max_col > self.tbl_version.get().version)
        )
        conn = Env.get().conn
        result = conn.execute(stmt).scalar_one()
        assert isinstance(result, int)
        return result

    def create(self) -> None:
        conn = Env.get().conn
        self.sa_md.create_all(bind=conn)

    def drop(self) -> None:
        """Drop store table"""
        conn = Env.get().conn
        self.sa_md.drop_all(bind=conn)

    def add_column(self, col: catalog.Column) -> None:
        """Add column(s) to the store-resident table based on a catalog column

        Note that a computed catalog column will require two extra columns (for the computed value and for the error
        message).
        """
        assert col.is_stored
        conn = Env.get().conn
        col_type_str = col.get_sa_col_type().compile(dialect=conn.dialect)
        s_txt = f'ALTER TABLE {self._storage_name()} ADD COLUMN {col.store_name()} {col_type_str} NULL'
        added_storage_cols = [col.store_name()]
        if col.stores_cellmd:
            cellmd_type_str = col.sa_cellmd_type().compile(dialect=conn.dialect)
            s_txt += f' , ADD COLUMN {col.cellmd_store_name()} {cellmd_type_str} DEFAULT NULL'
            added_storage_cols.append(col.cellmd_store_name())

        stmt = sql.text(s_txt)
        log_stmt(_logger, stmt)
        conn.execute(stmt)
        self.create_sa_tbl()
        _logger.info(f'Added columns {added_storage_cols} to storage table {self._storage_name()}')

    def drop_column(self, col: catalog.Column) -> None:
        """Execute Alter Table Drop Column statement"""
        s_txt = f'ALTER TABLE {self._storage_name()} DROP COLUMN {col.store_name()}'
        if col.stores_cellmd:
            s_txt += f' , DROP COLUMN {col.cellmd_store_name()}'
        stmt = sql.text(s_txt)
        log_stmt(_logger, stmt)
        Env.get().conn.execute(stmt)

    def ensure_columns_exist(self, cols: Iterable[catalog.Column]) -> None:
        conn = Env.get().conn
        sql_text = f'SELECT column_name FROM information_schema.columns WHERE table_name = {self._storage_name()!r}'
        result = conn.execute(sql.text(sql_text))
        existing_cols = {row[0] for row in result}
        for col in cols:
            if col.store_name() not in existing_cols:
                self.add_column(col)

    def load_column(self, col: catalog.Column, exec_plan: ExecNode, abort_on_exc: bool) -> int:
        """Update store column of a computed column with values produced by an execution plan

        Returns:
            number of rows with exceptions
        Raises:
            sql.exc.DBAPIError if there was a SQL error during execution
            excs.Error if on_error='abort' and there was an exception during row evaluation
        """
        assert col.tbl.id == self.tbl_version.id
        num_excs = 0
        num_rows = 0
        # create temp table to store output of exec_plan, with the same primary key as the store table
        tmp_name = f'temp_{self._storage_name()}'
        tmp_pk_cols = tuple(sql.Column(col.name, col.type, primary_key=True) for col in self.pk_columns())
        tmp_val_col_sql_idx = len(tmp_pk_cols)
        tmp_val_col = sql.Column(col.sa_col.name, col.sa_col.type)
        tmp_cols = [*tmp_pk_cols, tmp_val_col]
        # add error columns if the store column records errors
        if col.stores_cellmd:
            tmp_cellmd_col = sql.Column(col.sa_cellmd_col.name, col.sa_cellmd_col.type)
            tmp_cols.append(tmp_cellmd_col)
        tmp_col_names = [col.name for col in tmp_cols]

        tmp_tbl = sql.Table(tmp_name, self.sa_md, *tmp_cols, prefixes=['TEMPORARY'])
        conn = Env.get().conn
        tmp_tbl.create(bind=conn)

        row_builder = exec_plan.row_builder

        try:
            table_rows: list[tuple[Any]] = []

            # insert rows from exec_plan into temp table
            for row_batch in exec_plan:
                num_rows += len(row_batch)
                batch_table_rows: list[tuple[Any]] = []

                for row in row_batch:
                    if abort_on_exc and row.has_exc():
                        exc = row.get_first_exc()
                        raise excs.Error(f'Error while evaluating computed column {col.name!r}:\n{exc}') from exc
                    table_row, num_row_exc = row_builder.create_table_row(row, None, row.pk)
                    if col.col_type.is_media_type():
                        table_row[tmp_val_col_sql_idx] = self._move_tmp_media_file(
                            table_row[tmp_val_col_sql_idx], col, row.pk[-1]
                        )
                    num_excs += num_row_exc
                    batch_table_rows.append(tuple(table_row))

                table_rows.extend(batch_table_rows)

                if len(table_rows) >= self.__INSERT_BATCH_SIZE:
                    self.sql_insert(tmp_tbl, tmp_col_names, table_rows)
                    table_rows.clear()

            if len(table_rows) > 0:
                self.sql_insert(tmp_tbl, tmp_col_names, table_rows)

            # update store table with values from temp table
            update_stmt = sql.update(self.sa_tbl)
            for pk_col, tmp_pk_col in zip(self.pk_columns(), tmp_pk_cols):
                update_stmt = update_stmt.where(pk_col == tmp_pk_col)
            update_stmt = update_stmt.values({col.sa_col: tmp_val_col})
            if col.stores_cellmd:
                update_stmt = update_stmt.values({col.sa_cellmd_col: tmp_cellmd_col})
            log_explain(_logger, update_stmt, conn)
            conn.execute(update_stmt)

        finally:

            def remove_tmp_tbl() -> None:
                self.sa_md.remove(tmp_tbl)
                tmp_tbl.drop(bind=conn)

            run_cleanup(remove_tmp_tbl, raise_error=True)

        return num_excs

    def insert_rows(
        self,
        exec_plan: ExecNode,
        v_min: int,
        show_progress: bool = True,
        rowids: Optional[Iterator[int]] = None,
        abort_on_exc: bool = False,
    ) -> tuple[set[int], RowCountStats]:
        """Insert rows into the store table and update the catalog table's md
        Returns:
            number of inserted rows, number of exceptions, set of column ids that have exceptions
        """
        assert v_min is not None
        # TODO: total?
        num_excs = 0
        num_rows = 0
        cols_with_excs: set[int] = set()
        progress_bar: Optional[tqdm] = None  # create this only after we started executing
        row_builder = exec_plan.row_builder

        store_col_names, media_cols_by_idx = row_builder.store_column_names()

        try:
            table_rows: list[tuple[Any]] = []
            exec_plan.open()

            for row_batch in exec_plan:
                num_rows += len(row_batch)
                batch_table_rows: list[tuple[Any]] = []

                # compute batch of rows and convert them into table rows
                for row in row_batch:
                    # if abort_on_exc == True, we need to check for media validation exceptions
                    if abort_on_exc and row.has_exc():
                        exc = row.get_first_exc()
                        raise exc

                    rowid = (next(rowids),) if rowids is not None else row.pk[:-1]
                    pk = (*rowid, v_min)
                    assert len(pk) == len(self._pk_cols)
                    table_row, num_row_exc = row_builder.create_table_row(row, cols_with_excs, pk)
                    num_excs += num_row_exc

                    if show_progress:
                        if progress_bar is None:
                            warnings.simplefilter('ignore', category=TqdmWarning)
                            progress_bar = tqdm(
                                desc=f'Inserting rows into `{self.tbl_version.get().name}`',
                                unit=' rows',
                                ncols=100,
                                file=sys.stdout,
                            )
                        progress_bar.update(1)

                    self._move_tmp_media_files(table_row, media_cols_by_idx, v_min)
                    batch_table_rows.append(tuple(table_row))

                table_rows.extend(batch_table_rows)

                # if a batch is ready for insertion into the database, insert it
                if len(table_rows) >= self.__INSERT_BATCH_SIZE:
                    self.sql_insert(self.sa_tbl, store_col_names, table_rows)
                    table_rows.clear()

            # insert any remaining rows
            if len(table_rows) > 0:
                self.sql_insert(self.sa_tbl, store_col_names, table_rows)

            if progress_bar is not None:
                progress_bar.close()
            computed_values = exec_plan.ctx.num_computed_exprs * num_rows
            row_counts = RowCountStats(ins_rows=num_rows, num_excs=num_excs, computed_values=computed_values)

            return cols_with_excs, row_counts
        finally:
            exec_plan.close()

    @classmethod
    def sql_insert(cls, sa_tbl: sql.Table, store_col_names: list[str], table_rows: list[tuple[Any]]) -> None:
        assert len(table_rows) > 0
        conn = Env.get().conn
        conn.execute(sql.insert(sa_tbl), [dict(zip(store_col_names, table_row)) for table_row in table_rows])

        # TODO: Inserting directly via psycopg delivers a small performance benefit, but is somewhat fraught due to
        #     differences in the data representation that SQLAlchemy/psycopg expect. The below code will do the
        #     insertion in psycopg and can be used if/when we decide to pursue that optimization.
        # col_names_str = ", ".join(store_col_names)
        # placeholders_str = ", ".join('%s' for _ in store_col_names)
        # stmt_text = f'INSERT INTO {self.sa_tbl.name} ({col_names_str}) VALUES ({placeholders_str})'
        # conn.exec_driver_sql(stmt_text, table_rows)

    def _versions_clause(self, versions: list[Optional[int]], match_on_vmin: bool) -> sql.ColumnElement[bool]:
        """Return filter for base versions"""
        v = versions[0]
        if v is None:
            # we're looking at live rows
            clause = sql.and_(
                self.v_min_col <= self.tbl_version.get().version, self.v_max_col == schema.Table.MAX_VERSION
            )
        else:
            # we're looking at a specific version
            clause = self.v_min_col == v if match_on_vmin else self.v_max_col == v
        if len(versions) == 1:
            return clause
        return sql.and_(clause, self.base._versions_clause(versions[1:], match_on_vmin))

    def delete_rows(
        self,
        current_version: int,
        base_versions: list[Optional[int]],
        match_on_vmin: bool,
        where_clause: Optional[sql.ColumnElement[bool]],
    ) -> int:
        """Mark rows as deleted that are live and were created prior to current_version.
        Also: populate the undo columns
        Args:
            base_versions: if non-None, join only to base rows that were created at that version,
                otherwise join to rows that are live in the base's current version (which is distinct from the
                current_version parameter)
            match_on_vmin: if True, match exact versions on v_min; if False, match on v_max
            where_clause: if not None, also apply where_clause
        Returns:
            number of deleted rows
        """
        where_clause = sql.true() if where_clause is None else where_clause
        version_clause = sql.and_(self.v_min_col < current_version, self.v_max_col == schema.Table.MAX_VERSION)
        rowid_join_clause = self._rowid_join_predicate()
        base_versions_clause = (
            sql.true() if len(base_versions) == 0 else self.base._versions_clause(base_versions, match_on_vmin)
        )
        set_clause: dict[sql.Column, Union[int, sql.Column]] = {self.v_max_col: current_version}
        for index_info in self.tbl_version.get().idxs_by_name.values():
            # copy value column to undo column
            set_clause[index_info.undo_col.sa_col] = index_info.val_col.sa_col
            # set value column to NULL
            set_clause[index_info.val_col.sa_col] = None

        stmt = (
            sql.update(self.sa_tbl)
            .values(set_clause)
            .where(where_clause)
            .where(version_clause)
            .where(rowid_join_clause)
            .where(base_versions_clause)
        )
        conn = Env.get().conn
        log_explain(_logger, stmt, conn)
        status = conn.execute(stmt)
        return status.rowcount

    def dump_rows(self, version: int, filter_view: StoreBase, filter_view_version: int) -> Iterator[dict[str, Any]]:
        filter_predicate = sql.and_(
            filter_view.v_min_col <= filter_view_version,
            filter_view.v_max_col > filter_view_version,
            *[c1 == c2 for c1, c2 in zip(self.rowid_columns(), filter_view.rowid_columns())],
        )
        stmt = (
            sql.select('*')  # TODO: Use a more specific list of columns?
            .select_from(self.sa_tbl)
            .where(self.v_min_col <= version)
            .where(self.v_max_col > version)
            .where(sql.exists().where(filter_predicate))
        )
        conn = Env.get().conn
        _logger.debug(stmt)
        log_explain(_logger, stmt, conn)
        result = conn.execute(stmt)
        for row in result:
            yield dict(zip(result.keys(), row))

    def load_rows(self, rows: Iterable[dict[str, Any]], batch_size: int = 10_000) -> None:
        """
        When instantiating a replica, we can't rely on the usual insertion code path, which contains error handling
        and other logic that doesn't apply.
        """
        conn = Env.get().conn
        for batch in more_itertools.batched(rows, batch_size):
            conn.execute(sql.insert(self.sa_tbl), batch)


class StoreTable(StoreBase):
    def __init__(self, tbl_version: catalog.TableVersion):
        assert not tbl_version.is_view
        super().__init__(tbl_version)

    def _create_rowid_columns(self) -> list[sql.Column]:
        self.rowid_col = sql.Column('rowid', sql.BigInteger, nullable=False)
        return [self.rowid_col]

    def _storage_name(self) -> str:
        return f'tbl_{self.tbl_version.id.hex}'

    def _rowid_join_predicate(self) -> sql.ColumnElement[bool]:
        return sql.true()


class StoreView(StoreBase):
    def __init__(self, catalog_view: catalog.TableVersion):
        assert catalog_view.is_view
        super().__init__(catalog_view)

    def _create_rowid_columns(self) -> list[sql.Column]:
        # a view row corresponds directly to a single base row, which means it needs to duplicate its rowid columns
        self.rowid_cols = [sql.Column(c.name, c.type) for c in self.base.rowid_columns()]
        return self.rowid_cols

    def _storage_name(self) -> str:
        return f'view_{self.tbl_version.id.hex}'

    def _rowid_join_predicate(self) -> sql.ColumnElement[bool]:
        return sql.and_(
            self.base._rowid_join_predicate(),
            *[c1 == c2 for c1, c2 in zip(self.rowid_columns(), self.base.rowid_columns())],
        )


class StoreComponentView(StoreView):
    """A view that stores components of its base, as produced by a ComponentIterator

    PK: now also includes pos, the position returned by the ComponentIterator for the base row identified by base_rowid
    """

    rowid_cols: list[sql.Column]
    pos_col: sql.Column
    pos_col_idx: int

    def __init__(self, catalog_view: catalog.TableVersion):
        super().__init__(catalog_view)

    def _create_rowid_columns(self) -> list[sql.Column]:
        # each base row is expanded into n view rows
        self.rowid_cols = [sql.Column(c.name, c.type) for c in self.base.rowid_columns()]
        # name of pos column: avoid collisions with bases' pos columns
        self.pos_col = sql.Column(f'pos_{len(self.rowid_cols) - 1}', sql.BigInteger, nullable=False)
        self.pos_col_idx = len(self.rowid_cols)
        self.rowid_cols.append(self.pos_col)
        return self.rowid_cols

    def create_sa_tbl(self, tbl_version: Optional[catalog.TableVersion] = None) -> None:
        if tbl_version is None:
            tbl_version = self.tbl_version.get()
        super().create_sa_tbl(tbl_version)
        # we need to fix up the 'pos' column in TableVersion
        tbl_version.cols_by_name['pos'].sa_col = self.pos_col

    def _rowid_join_predicate(self) -> sql.ColumnElement[bool]:
        return sql.and_(
            self.base._rowid_join_predicate(),
            *[c1 == c2 for c1, c2 in zip(self.rowid_columns()[:-1], self.base.rowid_columns())],
        )
