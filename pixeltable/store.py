from __future__ import annotations

import abc
import logging
import os
import sys
import urllib.parse
import urllib.request
import warnings
from typing import Optional, Dict, Any, List, Tuple, Set

import sqlalchemy as sql
from tqdm import tqdm, TqdmWarning

import pixeltable.catalog as catalog
import pixeltable.env as env
from pixeltable import exprs
import pixeltable.exceptions as excs
from pixeltable.exec import ExecNode
from pixeltable.metadata import schema
from pixeltable.type_system import StringType
from pixeltable.utils.media_store import MediaStore
from pixeltable.utils.sql import log_stmt, log_explain

_logger = logging.getLogger('pixeltable')


class StoreBase:
    """Base class for stored tables

    Each row has the following system columns:
    - rowid columns: one or more columns that identify a user-visible row across all versions
    - v_min: version at which the row was created
    - v_max: version at which the row was deleted (or MAX_VERSION if it's still live)
    """

    def __init__(self, tbl_version: catalog.TableVersion):
        self.tbl_version = tbl_version
        self.sa_md = sql.MetaData()
        self.sa_tbl: Optional[sql.Table] = None
        self.create_sa_tbl()

    def pk_columns(self) -> List[sql.Column]:
        return self._pk_columns

    def rowid_columns(self) -> List[sql.Column]:
        return self._pk_columns[:-1]

    @abc.abstractmethod
    def _create_rowid_columns(self) -> List[sql.Column]:
        """Create and return rowid columns"""
        pass

    @abc.abstractmethod
    def _create_system_columns(self) -> List[sql.Column]:
        """Create and return system columns"""
        rowid_cols = self._create_rowid_columns()
        self.v_min_col = sql.Column('v_min', sql.BigInteger, nullable=False)
        self.v_max_col = \
            sql.Column('v_max', sql.BigInteger, nullable=False, server_default=str(schema.Table.MAX_VERSION))
        self._pk_columns = [*rowid_cols, self.v_min_col]
        return [*rowid_cols, self.v_min_col, self.v_max_col]


    def create_sa_tbl(self) -> None:
        """Create self.sa_tbl from self.tbl_version."""
        system_cols = self._create_system_columns()
        all_cols = system_cols.copy()
        idxs: List[sql.Index] = []
        for col in [c for c in self.tbl_version.cols if c.is_stored]:
            # re-create sql.Column for each column, regardless of whether it already has sa_col set: it was bound
            # to the last sql.Table version we created and cannot be reused
            col.create_sa_cols()
            all_cols.append(col.sa_col)
            if col.records_errors:
                all_cols.append(col.sa_errormsg_col)
                all_cols.append(col.sa_errortype_col)

            # we create an index for:
            # - scalar columns (except for strings, because long strings can't be used for B-tree indices)
            # - non-computed video and image columns (they will contain external paths/urls that users might want to
            #   filter on)
            if (col.col_type.is_scalar_type() and not col.col_type.is_string_type()) \
                    or (col.col_type.is_media_type() and not col.is_computed):
                # index names need to be unique within the Postgres instance
                idx_name = f'idx_{col.id}_{self.tbl_version.id.hex}'
                idxs.append(sql.Index(idx_name, col.sa_col))

        if self.sa_tbl is not None:
            # if we're called in response to a schema change, we need to remove the old table first
            self.sa_md.remove(self.sa_tbl)

        # index for all system columns:
        # - base x view joins can be executed as merge joins
        # - speeds up ORDER BY rowid DESC
        # - allows filtering for a particular table version in index scan
        idx_name = f'sys_cols_idx_{self.tbl_version.id.hex}'
        idxs.append(sql.Index(idx_name, *system_cols))
        # v_min/v_max indices: speeds up base table scans needed to propagate a base table insert or delete
        idx_name = f'vmin_idx_{self.tbl_version.id.hex}'
        idxs.append(sql.Index(idx_name, self.v_min_col, postgresql_using='brin'))
        idx_name = f'vmax_idx_{self.tbl_version.id.hex}'
        idxs.append(sql.Index(idx_name, self.v_max_col, postgresql_using='brin'))

        self.sa_tbl = sql.Table(self._storage_name(), self.sa_md, *all_cols, *idxs)

    @abc.abstractmethod
    def _rowid_join_predicate(self) -> sql.ClauseElement:
        """Return predicate for rowid joins to all bases"""
        pass

    @abc.abstractmethod
    def _storage_name(self) -> str:
        """Return the name of the data store table"""
        pass

    def _move_tmp_media_file(self, file_url: Optional[str], col: catalog.Column, v_min: int) -> str:
        """Move tmp media file with given url to Env.media_dir and return new url, or given url if not a tmp_dir file"""
        pxt_tmp_dir = str(env.Env.get().tmp_dir)
        if file_url is None:
            return None
        parsed = urllib.parse.urlparse(file_url)
        # We should never be passed a local file path here. The "len > 1" ensures that Windows
        # file paths aren't mistaken for URLs with a single-character scheme.
        assert len(parsed.scheme) > 1
        if parsed.scheme != 'file':
            # remote url
            return file_url
        file_path = urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
        if not file_path.startswith(pxt_tmp_dir):
            # not a tmp file
            return file_url
        _, ext = os.path.splitext(file_path)
        new_path = str(MediaStore.prepare_media_path(self.tbl_version.id, col.id, v_min, ext=ext))
        os.rename(file_path, new_path)
        new_file_url = urllib.parse.urljoin('file:', urllib.request.pathname2url(new_path))
        return new_file_url

    def _move_tmp_media_files(
            self, table_rows: List[Dict[str, Any]], media_cols: List[catalog.Column], v_min: int
    ) -> None:
        """Move tmp media files that we generated to a permanent location"""
        for c in media_cols:
            for table_row in table_rows:
                file_url = table_row[c.store_name()]
                table_row[c.store_name()] = self._move_tmp_media_file(file_url, c, v_min)

    def _create_table_row(
            self, input_row: exprs.DataRow, row_builder: exprs.RowBuilder, media_cols: List[catalog.Column],
            exc_col_ids: Set[int], v_min: int
    ) -> Tuple[Dict[str, Any], int]:
        """Return Tuple[complete table row, # of exceptions] for insert()
        Creates a row that includes the PK columns, with the values from input_row.pk.
        Returns:
            Tuple[complete table row, # of exceptions]
        """
        table_row, num_excs = row_builder.create_table_row(input_row, exc_col_ids)

        assert input_row.pk is not None and len(input_row.pk) == len(self._pk_columns)
        for pk_col, pk_val in zip(self._pk_columns, input_row.pk):
            if pk_col == self.v_min_col:
                table_row[pk_col.name] = v_min
            else:
                table_row[pk_col.name] = pk_val

        return table_row, num_excs

    def count(self, conn: Optional[sql.engine.Connection] = None) -> int:
        """Return the number of rows visible in self.tbl_version"""
        stmt = sql.select(sql.func.count('*'))\
            .select_from(self.sa_tbl)\
            .where(self.v_min_col <= self.tbl_version.version)\
            .where(self.v_max_col > self.tbl_version.version)
        if conn is None:
            with env.Env.get().engine.connect() as conn:
                result = conn.execute(stmt).scalar_one()
        else:
            result = conn.execute(stmt).scalar_one()
        assert isinstance(result, int)
        return result

    def create(self, conn: sql.engine.Connection) -> None:
        self.sa_md.create_all(bind=conn)

    def drop(self, conn: sql.engine.Connection) -> None:
        """Drop store table"""
        self.sa_md.drop_all(bind=conn)

    def add_column(self, col: catalog.Column, conn: sql.engine.Connection) -> None:
        """Add column(s) to the store-resident table based on a catalog column

        Note that a computed catalog column will require two extra columns (for the computed value and for the error
        message).
        """
        assert col.is_stored
        col_type_str = col.get_sa_col_type().compile(dialect=conn.dialect)
        stmt = sql.text(f'ALTER TABLE {self._storage_name()} ADD COLUMN {col.store_name()} {col_type_str} NULL')
        log_stmt(_logger, stmt)
        conn.execute(stmt)
        added_storage_cols = [col.store_name()]
        if col.records_errors:
            # we also need to create the errormsg and errortype storage cols
            stmt = (f'ALTER TABLE {self._storage_name()} '
                    f'ADD COLUMN {col.errormsg_store_name()} {StringType().to_sql()} DEFAULT NULL')
            conn.execute(sql.text(stmt))
            stmt = (f'ALTER TABLE {self._storage_name()} '
                    f'ADD COLUMN {col.errortype_store_name()} {StringType().to_sql()} DEFAULT NULL')
            conn.execute(sql.text(stmt))
            added_storage_cols.extend([col.errormsg_store_name(), col.errortype_store_name()])
        self.create_sa_tbl()
        _logger.info(f'Added columns {added_storage_cols} to storage table {self._storage_name()}')

    def drop_column(self, col: catalog.Column, conn: sql.engine.Connection) -> None:
        """Execute Alter Table Drop Column statement"""
        stmt = f'ALTER TABLE {self._storage_name()} DROP COLUMN {col.store_name()}'
        conn.execute(sql.text(stmt))
        if col.records_errors:
            stmt = f'ALTER TABLE {self._storage_name()} DROP COLUMN {col.errormsg_store_name()}'
            conn.execute(sql.text(stmt))
            stmt = f'ALTER TABLE {self._storage_name()} DROP COLUMN {col.errortype_store_name()}'
            conn.execute(sql.text(stmt))

    def load_column(
            self, col: catalog.Column, exec_plan: ExecNode, value_expr_slot_idx: int, conn: sql.engine.Connection
    ) -> int:
        """Update store column of a computed column with values produced by an execution plan

        Returns:
            number of rows with exceptions
        Raises:
            sql.exc.DBAPIError if there was an error during SQL execution
        """
        num_excs = 0
        num_rows = 0
        for row_batch in exec_plan:
            num_rows += len(row_batch)
            for result_row in row_batch:
                values_dict: Dict[sql.Column, Any] = {}

                if col.is_computed:
                    if result_row.has_exc(value_expr_slot_idx):
                        num_excs += 1
                        value_exc = result_row.get_exc(value_expr_slot_idx)
                        # we store a NULL value and record the exception/exc type
                        error_type = type(value_exc).__name__
                        error_msg = str(value_exc)
                        values_dict = {
                            col.sa_col: None,
                            col.sa_errortype_col: error_type,
                            col.sa_errormsg_col: error_msg
                        }
                    else:
                        val = result_row.get_stored_val(value_expr_slot_idx, col.sa_col.type)
                        if col.col_type.is_media_type():
                            val = self._move_tmp_media_file(val, col, result_row.pk[-1])
                        values_dict = {col.sa_col: val}

                update_stmt = sql.update(self.sa_tbl).values(values_dict)
                for pk_col, pk_val in zip(self.pk_columns(), result_row.pk):
                    update_stmt = update_stmt.where(pk_col == pk_val)
                log_stmt(_logger, update_stmt)
                conn.execute(update_stmt)

        return num_excs

    def insert_rows(
            self, exec_plan: ExecNode, conn: sql.engine.Connection, v_min: Optional[int] = None
    ) -> Tuple[int, int, Set[int]]:
        """Insert rows into the store table and update the catalog table's md
        Returns:
            number of inserted rows, number of exceptions, set of column ids that have exceptions
        """
        assert v_min is not None
        exec_plan.ctx.conn = conn
        batch_size = 16  # TODO: is this a good batch size?
        # TODO: total?
        num_excs = 0
        num_rows = 0
        cols_with_excs: Set[int] = set()
        progress_bar: Optional[tqdm] = None  # create this only after we started executing
        row_builder = exec_plan.row_builder
        media_cols = [info.col for info in row_builder.table_columns if info.col.col_type.is_media_type()]
        try:
            exec_plan.open()
            for row_batch in exec_plan:
                num_rows += len(row_batch)
                for batch_start_idx in range(0, len(row_batch), batch_size):
                    # compute batch of rows and convert them into table rows
                    table_rows: List[Dict[str, Any]] = []
                    for row_idx in range(batch_start_idx, min(batch_start_idx + batch_size, len(row_batch))):
                        row = row_batch[row_idx]
                        table_row, num_row_exc = \
                            self._create_table_row(row, row_builder, media_cols, cols_with_excs, v_min=v_min)
                        num_excs += num_row_exc
                        table_rows.append(table_row)
                        if progress_bar is None:
                            warnings.simplefilter("ignore", category=TqdmWarning)
                            progress_bar = tqdm(
                                desc=f'Inserting rows into `{self.tbl_version.name}`',
                                unit=' rows',
                                ncols=100,
                                file=sys.stdout
                            )
                        progress_bar.update(1)
                    self._move_tmp_media_files(table_rows, media_cols, v_min)
                    conn.execute(sql.insert(self.sa_tbl), table_rows)
            if progress_bar is not None:
                progress_bar.close()
            return num_rows, num_excs, cols_with_excs
        finally:
            exec_plan.close()

    def _versions_clause(self, versions: List[Optional[int]], match_on_vmin: bool) -> sql.ClauseElement:
        """Return filter for base versions"""
        v = versions[0]
        if v is None:
            # we're looking at live rows
            clause = sql.and_(self.v_min_col <= self.tbl_version.version, self.v_max_col == schema.Table.MAX_VERSION)
        else:
            # we're looking at a specific version
            clause = self.v_min_col == v if match_on_vmin else self.v_max_col == v
        if len(versions) == 1:
            return clause
        return sql.and_(clause, self.base._versions_clause(versions[1:], match_on_vmin))

    def delete_rows(
            self, current_version: int, base_versions: List[Optional[int]], match_on_vmin: bool,
            where_clause: Optional[sql.ClauseElement], conn: sql.engine.Connection) -> int:
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
        where_clause = sql.and_(
            self.v_min_col < current_version,
            self.v_max_col == schema.Table.MAX_VERSION,
            where_clause)
        rowid_join_clause = self._rowid_join_predicate()
        base_versions_clause = sql.true() if len(base_versions) == 0 \
            else self.base._versions_clause(base_versions, match_on_vmin)
        set_clause = {self.v_max_col: current_version}
        for index_info in self.tbl_version.idxs_by_name.values():
            # copy value column to undo column
            set_clause[index_info.undo_col.sa_col] = index_info.val_col.sa_col
            # set value column to NULL
            set_clause[index_info.val_col.sa_col] = None
        stmt = sql.update(self.sa_tbl) \
            .values(set_clause) \
            .where(where_clause) \
            .where(rowid_join_clause) \
            .where(base_versions_clause)
        log_explain(_logger, stmt, conn)
        status = conn.execute(stmt)
        return status.rowcount


class StoreTable(StoreBase):
    def __init__(self, tbl_version: catalog.TableVersion):
        assert not tbl_version.is_view()
        super().__init__(tbl_version)

    def _create_rowid_columns(self) -> List[sql.Column]:
        self.rowid_col = sql.Column('rowid', sql.BigInteger, nullable=False)
        return [self.rowid_col]

    def _storage_name(self) -> str:
        return f'tbl_{self.tbl_version.id.hex}'

    def _rowid_join_predicate(self) -> sql.ClauseElement:
        return sql.true()


class StoreView(StoreBase):
    def __init__(self, catalog_view: catalog.TableVersion):
        assert catalog_view.is_view()
        self.base = catalog_view.base.store_tbl
        super().__init__(catalog_view)

    def _create_rowid_columns(self) -> List[sql.Column]:
        # a view row corresponds directly to a single base row, which means it needs to duplicate its rowid columns
        self.rowid_cols = [sql.Column(c.name, c.type) for c in self.base.rowid_columns()]
        return self.rowid_cols

    def _storage_name(self) -> str:
        return f'view_{self.tbl_version.id.hex}'

    def _rowid_join_predicate(self) -> sql.ClauseElement:
        return sql.and_(
            self.base._rowid_join_predicate(),
            *[c1 == c2 for c1, c2 in zip(self.rowid_columns(), self.base.rowid_columns())])

class StoreComponentView(StoreView):
    """A view that stores components of its base, as produced by a ComponentIterator

    PK: now also includes pos, the position returned by the ComponentIterator for the base row identified by base_rowid
    """
    def __init__(self, catalog_view: catalog.TableVersion):
        super().__init__(catalog_view)

    def _create_rowid_columns(self) -> List[sql.Column]:
        # each base row is expanded into n view rows
        self.rowid_cols = [sql.Column(c.name, c.type) for c in self.base.rowid_columns()]
        # name of pos column: avoid collisions with bases' pos columns
        self.pos_col = sql.Column(f'pos_{len(self.rowid_cols) - 1}', sql.BigInteger, nullable=False)
        self.pos_col_idx = len(self.rowid_cols)
        self.rowid_cols.append(self.pos_col)
        return self.rowid_cols

    def create_sa_tbl(self) -> None:
        super().create_sa_tbl()
        # we need to fix up the 'pos' column in TableVersion
        self.tbl_version.cols_by_name['pos'].sa_col = self.pos_col

    def _rowid_join_predicate(self) -> sql.ClauseElement:
        return sql.and_(
            self.base._rowid_join_predicate(),
            *[c1 == c2 for c1, c2 in zip(self.rowid_columns()[:-1], self.base.rowid_columns())])
