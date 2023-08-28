from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, Set
import logging
import dataclasses
import abc

import sqlalchemy as sql
from tqdm.autonotebook import tqdm

from pixeltable import catalog
from pixeltable.metadata import schema
from pixeltable.type_system import StringType
from pixeltable.exec import ExecNode, DataRowBatch, ColumnInfo
from pixeltable import exprs


_logger = logging.getLogger('pixeltable')


class StoreBase:
    """Base class for stored tables"""
    def __init__(self, tbl_version: catalog.TableVersion):
        self.tbl_version = tbl_version
        self.sa_md = sql.MetaData()
        self.sa_tbl: Optional[sql.Table] = None
        self._create_sa_tbl()

    @abc.abstractmethod
    def pk_columns(self) -> List[sql.Column]:
        """Return primary key columns"""
        pass

    @abc.abstractmethod
    def _create_system_columns(self) -> List[sql.Column]:
        """Create and return system columns"""
        pass

    def _create_sa_tbl(self) -> None:
        """Create self.sa_tbl from self.tbl_version."""
        store_cols = self._create_system_columns()
        for col in [c for c in self.tbl_version.cols if c.is_stored]:
            # re-create sql.Column for each column, regardless of whether it already has sa_col set: it was bound
            # to the last sql.Table version we created and cannot be reused
            col.create_sa_cols()
            store_cols.append(col.sa_col)
            if col.is_computed:
                store_cols.append(col.sa_errormsg_col)
                store_cols.append(col.sa_errortype_col)
            if col.is_indexed:
                store_cols.append(col.sa_idx_col)

        if self.sa_tbl is not None:
            # if we're called in response to a schema change, we need to remove the old table first
            self.sa_md.remove(self.sa_tbl)
        self.sa_tbl = sql.Table(self._storage_name(), self.sa_md, *store_cols)

    @abc.abstractmethod
    def _storage_name(self) -> str:
        """Return the name of the data store table"""
        pass

    def _create_row(
            self, input_row: exprs.DataRow, schema_col_info: List[ColumnInfo], idx_col_info: List[ColumnInfo],
            exc_col_ids: Set[int]
    ) -> Tuple[Dict[str, Any], int]:
        """Return Tuple[dict that represents a stored row (can be passed to sql.insert()), # of exceptions]
            This excludes system columns.
        """
        num_excs = 0
        table_row: Dict[str, Any] = {}
        for info in schema_col_info:
            if input_row.has_exc(info.slot_idx):
                # exceptions get stored in the errortype/-msg columns
                exc = input_row.get_exc(info.slot_idx)
                num_excs += 1
                exc_col_ids.add(info.col.id)
                table_row[info.col.storage_name()] = None
                table_row[info.col.errortype_storage_name()] = type(exc).__name__
                table_row[info.col.errormsg_storage_name()] = str(exc)
            else:
                val = input_row.get_stored_val(info.slot_idx)
                table_row[info.col.storage_name()] = val
                # we unfortunately need to set these, even if there are no errors
                table_row[info.col.errortype_storage_name()] = None
                table_row[info.col.errormsg_storage_name()] = None

        for info in idx_col_info:
            # don't use get_stored_val() here, we need to pass in the ndarray
            val = input_row[info.slot_idx]
            table_row[info.col.index_storage_name()] = val

        return table_row, num_excs

    @abc.abstractmethod
    def _create_insert_row(
        self, input_row: exprs.DataRow, schema_col_info: List[ColumnInfo], idx_col_info: List[ColumnInfo],
        exc_col_ids: Set[int]
    ) -> Tuple[Dict[str, Any], int]:
        """Return Tuple[complete table row, # of exceptions] for insert()"""
        pass

    @abc.abstractmethod
    def _create_update_row(
            self, input_row: exprs.DataRow, schema_col_info: List[ColumnInfo], idx_col_info: List[ColumnInfo],
            exc_col_ids: Set[int]
    ) -> Tuple[Dict[str, Any], int]:
        """Return Tuple[complete table row, # of exceptions] for update()"""
        pass

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
        stmt = f'ALTER TABLE {self._storage_name()} ADD COLUMN {col.storage_name()} {col.col_type.to_sql()}'
        conn.execute(sql.text(stmt))
        added_storage_cols = [col.storage_name()]
        if col.is_computed:
            # we also need to create the errormsg and errortype storage cols
            stmt = (f'ALTER TABLE {self._storage_name()} '
                    f'ADD COLUMN {col.errormsg_storage_name()} {StringType().to_sql()} DEFAULT NULL')
            conn.execute(sql.text(stmt))
            stmt = (f'ALTER TABLE {self._storage_name()} '
                    f'ADD COLUMN {col.errortype_storage_name()} {StringType().to_sql()} DEFAULT NULL')
            conn.execute(sql.text(stmt))
            added_storage_cols.extend([col.errormsg_storage_name(), col.errortype_storage_name()])
        self._create_sa_tbl()
        _logger.info(f'Added columns {added_storage_cols} to storage table {self._storage_name()}')

    def drop_column(self, col: Optional[catalog.Column] = None, conn: Optional[sql.engine.Connection] = None) -> None:
        """Re-create self.sa_tbl and drop column, if one is given"""
        if col is not None:
            assert conn is not None
            stmt = f'ALTER TABLE {self._storage_name()} DROP COLUMN {col.storage_name()}'
            conn.execute(sql.text(stmt))
            if col.is_computed:
                stmt = f'ALTER TABLE {self._storage_name()} DROP COLUMN {col.errormsg_storage_name()}'
                conn.execute(sql.text(stmt))
                stmt = f'ALTER TABLE {self._storage_name()} DROP COLUMN {col.errortype_storage_name()}'
                conn.execute(sql.text(stmt))
        self._create_sa_tbl()

    def load_column(
            self, col: catalog.Column, exec_plan: ExecNode, value_expr_slot_idx: int, embedding_slot_idx: int,
            conn: sql.engine.Connection
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
                        val = result_row.get_stored_val(value_expr_slot_idx)
                        values_dict = {col.sa_col: val}

                if col.is_indexed:
                    # TODO: deal with exceptions
                    assert not result_row.has_exc(embedding_slot_idx)
                    # don't use get_stored_val() here, we need to pass the ndarray
                    embedding = result_row[embedding_slot_idx]
                    values_dict[col.sa_index_col] = embedding

                update_stmt = sql.update(self.sa_tbl).values(values_dict)
                for pk_col, pk_val in zip(self.pk_columns(), result_row.pk):
                    update_stmt = update_stmt.where(pk_col == pk_val)
                conn.execute(update_stmt)

        return num_excs

    def insert_rows(
            self, exec_plan: ExecNode, schema_col_info: List[ColumnInfo], idx_col_info: List[ColumnInfo],
            conn: sql.engine.Connection
    ) -> Tuple[int, int, Set[int]]:
        """Insert rows into the store table and update the catalog table's md
        Returns:
            number of inserted rows, number of exceptions, set of column ids that have exceptions
        """
        exec_plan.ctx.conn = conn
        batch_size = 16  # TODO: is this a good batch size?
        # TODO: total?
        num_excs = 0
        num_rows = 0
        cols_with_excs: Set[int] = set()
        progress_bar: Optional[tqdm] = None  # create this only after we started executing
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
                            self._create_insert_row(row, schema_col_info, idx_col_info, cols_with_excs)
                        num_excs += num_row_exc
                        table_rows.append(table_row)
                        if progress_bar is None:
                            progress_bar = tqdm(desc='Inserting rows into table', unit='rows')
                        progress_bar.update(1)
                    conn.execute(sql.insert(self.sa_tbl), table_rows)
            if progress_bar is not None:
                progress_bar.close()
            return num_rows, num_excs, cols_with_excs
        finally:
            exec_plan.close()

    def update_rows(
            self, exec_plan: ExecNode, row_info: List[ColumnInfo], where_clause: Optional[sql.sql.ClauseElement],
            conn: sql.engine.Connection
    ) -> Tuple[int, int, Set[int]]:
        """Update rows in the store table
        Returns:
            number of rows, number of exceptions, set of column ids that have exceptions
        """
        exec_plan.ctx.conn = conn
        num_excs = 0
        num_rows = 0
        cols_with_excs: Set[int] = set()
        try:
            # insert new versions of updated rows
            for row_batch in exec_plan:
                num_rows += len(row_batch)
                table_rows: List[Dict[str, Any]] = []
                for result_row in row_batch:
                    # idx_col_info=[]: we assume that embeddings don't change
                    table_row, num_row_exc = self._create_update_row(result_row, row_info, [], cols_with_excs)
                    num_excs += num_row_exc
                    table_rows.append(table_row)
                conn.execute(sql.insert(self.sa_tbl), table_rows)
        finally:
            exec_plan.close()

        # mark old versions (v_min < self.version) of updated rows as deleted
        where_clause = where_clause if where_clause is not None else sql.true()
        stmt = sql.update(self.sa_tbl) \
            .values({self.v_max_col: self.tbl_version.version}) \
            .where(self.v_min_col < self.tbl_version.version) \
            .where(self.v_max_col == schema.Table.MAX_VERSION) \
            .where(where_clause)
        conn.execute(stmt)

        return num_rows, num_excs, cols_with_excs


class StoreTable(StoreBase):
    def __init__(self, tbl_version: catalog.TableVersion):
        assert not tbl_version.is_view()
        super().__init__(tbl_version)

    def pk_columns(self) -> List[sql.Column]:
        return self._pk_columns

    def _create_system_columns(self) -> List[sql.Column]:
        self.rowid_col = sql.Column('rowid', sql.BigInteger, nullable=False)
        self.v_min_col = sql.Column('v_min', sql.BigInteger, nullable=False)
        self.v_max_col = \
            sql.Column('v_max', sql.BigInteger, nullable=False, server_default=str(schema.Table.MAX_VERSION))
        self._pk_columns = [self.rowid_col, self.v_min_col]
        return [self.rowid_col, self.v_min_col, self.v_max_col]

    def _storage_name(self) -> str:
        return f'tbl_{self.tbl_version.id.hex}'

    def _create_insert_row(
            self, input_row: exprs.DataRow, schema_col_info: List[ColumnInfo], idx_col_info: List[ColumnInfo],
            exc_col_ids: Set[int]
    ) -> Tuple[Dict[str, Any], int]:
        """Create a row with a new rowid and the current table version
        Returns:
             Tuple[complete table row, # of exceptions]
         """
        table_row, num_excs = self._create_row(input_row, schema_col_info, idx_col_info, exc_col_ids)
        table_row.update({
            self.rowid_col.name: self.tbl_version.next_rowid,
            self.v_min_col.name: self.tbl_version.version,
        })
        self.tbl_version.next_rowid += 1
        return table_row, num_excs

    def _create_update_row(
            self, input_row: exprs.DataRow, schema_col_info: List[ColumnInfo], idx_col_info: List[ColumnInfo],
            exc_col_ids: Set[int]
    ) -> Tuple[Dict[str, Any], int]:
        """Create a row with the same rowid as the input and the current table version
        Returns:
            Tuple[complete table row, # of exceptions]
        """
        table_row, num_excs = self._create_row(input_row, schema_col_info, idx_col_info, exc_col_ids)
        assert input_row.pk is not None and len(input_row.pk) == 2
        table_row.update({
            self.rowid_col.name: input_row.pk[0],
            self.v_min_col.name: self.tbl_version.version,
        })
        return table_row, num_excs


class StoreView(StoreBase):
    def __init__(self, catalog_view: catalog.TableVersion):
        assert catalog_view.is_view()
        super().__init__(catalog_view)

    def pk_columns(self) -> List[sql.Column]:
        return self._pk_columns

    def _create_system_columns(self) -> List[sql.Column]:
        self.base_rowid_col = sql.Column('base_rowid', sql.BigInteger, nullable=False)
        self.base_v_min_col = sql.Column('base_v_min', sql.BigInteger, nullable=False)
        self.base_v_max_col = \
            sql.Column('base_v_max', sql.BigInteger, nullable=False, server_default=str(schema.Table.MAX_VERSION))
        self.v_min_col = sql.Column('v_min', sql.BigInteger, nullable=False)
        self.v_max_col = \
            sql.Column('v_max', sql.BigInteger, nullable=False, server_default=str(schema.Table.MAX_VERSION))
        self._pk_columns = [self.base_rowid_col, self.base_v_min_col, self.v_min_col]
        return [self.base_rowid_col, self.base_v_min_col, self.base_v_max_col, self.v_min_col, self.v_max_col]

    def _storage_name(self) -> str:
        return f'view_{self.tbl_version.id.hex}'

    def _create_insert_row(
            self, input_row: exprs.DataRow, schema_col_info: List[ColumnInfo], idx_col_info: List[ColumnInfo],
            exc_col_ids: Set[int]
    ) -> Tuple[Dict[str, Any], int]:
        """Creates a row with the input's rowid/v_min and the current table version
        Returns:
            Tuple[complete table row, # of exceptions]
        """
        table_row, num_excs = self._create_row(input_row, schema_col_info, idx_col_info, exc_col_ids)
        # the input row is from the base table
        assert input_row.pk is not None and len(input_row.pk) == 2
        table_row.update({
            self.base_rowid_col.name: input_row.pk[0],
            self.base_v_min_col.name: input_row.pk[1],
            self.v_min_col.name: self.tbl_version.version,
        })
        return table_row, num_excs

    def _create_update_row(
            self, input_row: exprs.DataRow, schema_col_info: List[ColumnInfo], idx_col_info: List[ColumnInfo],
            exc_col_ids: Set[int]
    ) -> Tuple[Dict[str, Any], int]:
        """Return Tuple[complete table row, # of exceptions] for update()"""
        result = self._create_row(input_row, schema_col_info, idx_col_info, exc_col_ids)
        # the input row is from this view
        assert input_row.pk is not None and len(input_row.pk) == 3
        result.update({
            self.base_rowid_col.name: input_row.pk[0],
            self.base_v_min_col.name: input_row.pk[1],
            self.v_min_col.name: self.tbl_version.version,
        })
        return result

    def mark_deleted(self, base_version: int, conn: sql.engine.Connection) -> None:
        """Mark rows that were superseded by a new base table version as deleted"""
        v = self.sa_tbl.alias('v')
        # we use a self-join to find rows that were superseded by a new base table version:
        # - new rows have base_v_min == base_version
        # - old rows have base_v_min < base_version
        # - old rows are visible (v_min <= self.version && v_max == MAX_VERSION)
        stmt = sql.update(self.sa_tbl) \
            .values({self.base_v_max_col: base_version}) \
            .where(self.base_rowid_col == v.c.base_rowid) \
            .where(v.c.base_v_min == base_version) \
            .where(self.base_v_min_col < base_version) \
            .where(self.base_v_max_col == schema.Table.MAX_VERSION) \
            .where(self.v_min_col <= self.tbl_version.version) \
            .where(self.v_max_col == schema.Table.MAX_VERSION)
        conn.execute(stmt)
