from __future__ import annotations

import logging
import time
from typing import Optional, Union, Any
from uuid import UUID

import sqlalchemy as sql

import pixeltable
import pixeltable.exceptions as excs
from pixeltable import exprs
from pixeltable.env import Env
from .column import Column
from .globals import POS_COLUMN_NAME, _ROWID_COLUMN_NAME, UpdateStatus
from .table_version import TableVersion

_logger = logging.getLogger('pixeltable')


class TableVersionPath:
    """
    A TableVersionPath represents the sequence of TableVersions from a base table to a particular view:
    - for a base table: only includes that TableVersion
    - for a view: includes the TableVersion for the view and all its bases
    - multiple snapshots can reference the same TableVersion, but with different bases, which means that the
      graph of TableVersions is a DAG, not a tree (which is why we cannot embed the DAG into TableVersion directly)

    TableVersionPath contains all metadata needed to execute queries and updates against a particular version of a
    table/view.
    """

    def __init__(self, tbl_version: TableVersion, base: Optional[TableVersionPath] = None):
        assert tbl_version is not None
        self.tbl_version = tbl_version
        self.base = base

    def tbl_id(self) -> UUID:
        """Return the id of the table/view that this path represents"""
        return self.tbl_version.id

    def version(self) -> int:
        """Return the version of the table/view that this path represents"""
        return self.tbl_version.version

    def tbl_name(self) -> str:
        """Return the name of the table/view that this path represents"""
        return self.tbl_version.name

    def path_len(self) -> int:
        """Return the length of the path"""
        return 1 if self.base is None else 1 + self.base.path_len()

    def is_snapshot(self) -> bool:
        """Return True if this is a path of snapshot versions"""
        if not self.tbl_version.is_snapshot:
            return False
        return self.base.is_snapshot() if self.base is not None else True

    def is_view(self) -> bool:
        return self.tbl_version.is_view()

    def is_component_view(self) -> bool:
        return self.tbl_version.is_component_view()

    def is_insertable(self) -> bool:
        return self.tbl_version.is_insertable()

    def get_tbl_versions(self) -> list[TableVersion]:
        """Return all tbl versions"""
        if self.base is None:
            return [self.tbl_version]
        return [self.tbl_version] + self.base.get_tbl_versions()

    def get_bases(self) -> list[TableVersion]:
        """Return all tbl versions"""
        if self.base is None:
            return []
        return self.base.get_tbl_versions()

    def find_tbl_version(self, id: UUID) -> Optional[TableVersion]:
        """Return the matching TableVersion in the chain of TableVersions, starting with this one"""
        if self.tbl_version.id == id:
            return self.tbl_version
        if self.base is None:
            return None
        return self.base.find_tbl_version(id)

    def __getattr__(self, col_name: str) -> 'pixeltable.exprs.ColumnRef':
        """Return a ColumnRef for the given column name."""
        from pixeltable.exprs import ColumnRef, RowidRef
        if col_name == POS_COLUMN_NAME and self.is_component_view():
            return RowidRef(self.tbl_version, self.tbl_version.store_tbl.pos_col_idx)
        if col_name not in self.tbl_version.cols_by_name:
            if self.base is None:
                raise AttributeError(f'Column {col_name} unknown')
            return getattr(self.base, col_name)
        col = self.tbl_version.cols_by_name[col_name]
        return ColumnRef(col)

    def __getitem__(self, index: object) -> Union['pixeltable.exprs.ColumnRef', 'pixeltable.dataframe.DataFrame']:
        """Return a ColumnRef for the given column name, or a DataFrame for the given slice.
        """
        if isinstance(index, str):
            # basically <tbl>.<colname>
            return self.__getattr__(index)
        from pixeltable.dataframe import DataFrame
        return DataFrame(self).__getitem__(index)

    def columns(self) -> list[Column]:
        """Return all user columns visible in this tbl version path, including columns from bases"""
        result = list(self.tbl_version.cols_by_name.values())
        if self.base is not None:
            base_cols = self.base.columns()
            # we only include base columns that don't conflict with one of our column names
            result.extend(c for c in base_cols if c.name not in self.tbl_version.cols_by_name)
        return result

    def cols_by_name(self) -> dict[str, Column]:
        """Return a dict of all user columns visible in this tbl version path, including columns from bases"""
        cols = self.columns()
        return {col.name: col for col in cols}

    def cols_by_id(self) -> dict[int, Column]:
        """Return a dict of all user columns visible in this tbl version path, including columns from bases"""
        cols = self.columns()
        return {col.id: col for col in cols}

    def get_column(self, name: str, include_bases: bool = True) -> Optional[Column]:
        """Return the column with the given name, or None if not found"""
        col = self.tbl_version.cols_by_name.get(name)
        if col is not None:
            return col
        elif self.base is not None and include_bases:
            return self.base.get_column(name)
        else:
            return None

    def has_column(self, col: Column, include_bases: bool = True) -> bool:
        """Return True if this table has the given column.
        """
        assert col.tbl is not None
        if col.tbl.id == self.tbl_version.id and col.tbl.effective_version == self.tbl_version.effective_version \
                and col.id in self.tbl_version.cols_by_id:
            # the column is visible in this table version
            return True
        elif self.base is not None and include_bases:
            return self.base.has_column(col)
        else:
            return False

    def update(
        self, value_spec: dict[str, Any], where: Optional['pixeltable.exprs.Predicate'] = None, cascade: bool = True
    ) -> UpdateStatus:
        """Update rows in this TableVersionPath.
        Args:
            value_spec: a list of (column, value) pairs specifying the columns to update and their new values.
            where: a Predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns,
                including within views.
        """
        if self.is_snapshot():
            raise excs.Error('Cannot update a snapshot')

        from pixeltable.plan import Planner

        update_spec = self._validate_update_spec(value_spec, allow_pk=False, allow_exprs=True)
        if where is not None:
            if not isinstance(where, exprs.Predicate):
                raise excs.Error(f"'where' argument must be a Predicate, got {type(where)}")
            analysis_info = Planner.analyze(self, where)
            # for now we require that the updated rows can be identified via SQL, rather than via a Python filter
            if analysis_info.filter is not None:
                raise excs.Error(f'Filter {analysis_info.filter} not expressible in SQL')

        with Env.get().engine.begin() as conn:
            return self._update(conn, update_spec, where, cascade)

    def batch_update(
            self, batch: list[dict[Column, 'pixeltable.exprs.Expr']], rowids: list[tuple[int, ...]],
            cascade: bool = True
    ) -> UpdateStatus:
        """Update rows in batch.
        Args:
            batch: one dict per row, each mapping Columns to LiteralExprs representing the new values
            rowids: if not empty, one tuple per row, each containing the rowid values for the corresponding row in batch
        """
        # if we do lookups of rowids, we must have one for each row in the batch
        assert len(rowids) == 0 or len(rowids) == len(batch)
        result_status = UpdateStatus()
        cols_with_excs: set[str] = set()
        updated_cols: set[str] = set()
        pk_cols = self.tbl_version.primary_key_columns()
        use_rowids = len(rowids) > 0

        with Env.get().engine.begin() as conn:
            for i, row in enumerate(batch):
                where_clause: Optional[exprs.Expr] = None
                if use_rowids:
                    # construct Where clause to match rowid
                    num_rowid_cols = len(self.tbl_version.store_tbl.rowid_columns())
                    for col_idx in range(num_rowid_cols):
                        assert len(rowids[i]) == num_rowid_cols, f'len({rowids[i]}) != {num_rowid_cols}'
                        clause = exprs.RowidRef(self.tbl_version, col_idx) == rowids[i][col_idx]
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
        from pixeltable.plan import Planner

        plan, updated_cols, recomputed_cols = (
            Planner.create_update_plan(self, update_targets, [], where_clause, cascade)
        )
        result = self.tbl_version.propagate_update(
            plan, where_clause.sql_expr() if where_clause is not None else None, recomputed_cols,
            base_versions=[], conn=conn, timestamp=time.time(), cascade=cascade, show_progress=show_progress)
        result.updated_cols = updated_cols
        return result

    def _validate_update_spec(
            self, value_spec: dict[str, Any], allow_pk: bool, allow_exprs: bool
    ) -> dict[Column, 'pixeltable.exprs.Expr']:
        update_targets: dict[Column, exprs.Expr] = {}
        for col_name, val in value_spec.items():
            if not isinstance(col_name, str):
                raise excs.Error(f'Update specification: dict key must be column name, got {col_name!r}')
            if col_name == _ROWID_COLUMN_NAME:
                # ignore pseudo-column _rowid
                continue
            col = self.get_column(col_name, include_bases=False)
            if col is None:
                # TODO: return more informative error if this is trying to update a base column
                raise excs.Error(f'Column {col_name} unknown')
            if col.is_computed:
                raise excs.Error(f'Column {col_name} is computed and cannot be updated')
            if col.is_pk and not allow_pk:
                raise excs.Error(f'Column {col_name} is a primary key column and cannot be updated')
            if col.col_type.is_media_type():
                raise excs.Error(f'Column {col_name} has type image/video/audio/document and cannot be updated')

            # make sure that the value is compatible with the column type
            try:
                # check if this is a literal
                value_expr = exprs.Literal(val, col_type=col.col_type)
            except TypeError:
                if not allow_exprs:
                    raise excs.Error(
                        f'Column {col_name}: value {val!r} is not a valid literal for this column '
                        f'(expected {col.col_type})')
                # it's not a literal, let's try to create an expr from it
                value_expr = exprs.Expr.from_object(val)
                if value_expr is None:
                    raise excs.Error(f'Column {col_name}: value {val!r} is not a recognized literal or expression')
                if not col.col_type.matches(value_expr.col_type):
                    raise excs.Error((
                        f'Type of value {val!r} ({value_expr.col_type}) is not compatible with the type of column '
                        f'{col_name} ({col.col_type})'
                    ))
            update_targets[col] = value_expr

        return update_targets

    def delete(self, where: Optional['pixeltable.exprs.Predicate'] = None) -> UpdateStatus:
        """Delete rows in this table.
        Args:
            where: a Predicate to filter rows to delete.
        """
        assert self.is_insertable()
        from pixeltable.exprs import Predicate
        from pixeltable.plan import Planner
        if where is not None:
            if not isinstance(where, Predicate):
                raise excs.Error(f"'where' argument must be a Predicate, got {type(where)}")
            analysis_info = Planner.analyze(self, where)
            # for now we require that the updated rows can be identified via SQL, rather than via a Python filter
            if analysis_info.filter is not None:
                raise excs.Error(f'Filter {analysis_info.filter} not expressible in SQL')

        analysis_info = Planner.analyze(self, where)
        with Env.get().engine.begin() as conn:
            num_rows = self.tbl_version.propagate_delete(analysis_info.sql_where_clause, base_versions=[], conn=conn, timestamp=time.time())

        status = UpdateStatus(num_rows=num_rows)
        return status

    def as_dict(self) -> dict:
        return {
            'tbl_version': self.tbl_version.as_dict(),
            'base': self.base.as_dict() if self.base is not None else None
        }

    @classmethod
    def from_dict(cls, d: dict) -> TableVersionPath:
        tbl_version = TableVersion.from_dict(d['tbl_version'])
        base = TableVersionPath.from_dict(d['base']) if d['base'] is not None else None
        return cls(tbl_version, base)
