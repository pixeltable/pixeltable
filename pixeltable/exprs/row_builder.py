from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence
from uuid import UUID

import numpy as np

from pixeltable import catalog, exceptions as excs, utils
from pixeltable.env import Env
from pixeltable.utils.media_store import MediaStore

from .data_row import DataRow
from .expr import Expr, ExprScope
from .expr_set import ExprSet


class ExecProfile:
    def __init__(self, row_builder: RowBuilder):
        self.eval_time = [0.0] * row_builder.num_materialized
        self.eval_count = [0] * row_builder.num_materialized
        self.row_builder = row_builder

    def print(self, num_rows: int) -> None:
        for i in range(self.row_builder.num_materialized):
            if self.eval_count[i] == 0:
                continue
            per_call_time = self.eval_time[i] / self.eval_count[i]
            calls_per_row = self.eval_count[i] / num_rows
            multiple_str = f'({calls_per_row}x)' if calls_per_row > 1 else ''
            Env.get().console_logger.info(
                f'{self.row_builder.unique_exprs[i]}: {utils.print_perf_counter_delta(per_call_time)} {multiple_str}'
            )


@dataclass
class ColumnSlotIdx:
    """Info for how to locate materialized column in DataRow
    TODO: can this be integrated into RowBuilder directly?
    """

    col: catalog.Column
    slot_idx: int


class RowBuilder:
    """Create and populate DataRows and table rows from exprs and computed columns

    For ColumnRefs to unstored iterator columns:
    - in order for them to be executable, we also record the iterator args and pass them to the ColumnRef
    """

    unique_exprs: ExprSet
    next_slot_idx: int
    input_expr_slot_idxs: set[int]

    # output exprs: all exprs the caller wants to materialize
    # - explicitly requested output_exprs
    # - values for computed columns
    output_exprs: ExprSet

    input_exprs: ExprSet

    tbl: Optional[catalog.TableVersion]  # reference table of the RowBuilder; used to identify pk columns for writes
    table_columns: list[ColumnSlotIdx]
    default_eval_ctx: EvalCtx
    unstored_iter_args: dict[UUID, Expr]

    # transitive dependents for the purpose of exception propagation: an exception for slot i is propagated to
    # _exc_dependents[i]
    # (list of set of slot_idxs, indexed by slot_idx)
    _exc_dependents: list[set[int]]

    # dependents[i] = direct dependents of expr with slot idx i; dependents[i, j] == True: expr j depends on expr i
    dependents: np.ndarray  # of bool
    transitive_dependents: np.ndarray  # of bool
    # dependencies[i] = direct dependencies of expr with slot idx i; transpose of dependents
    dependencies: np.ndarray  # of bool
    # num_dependencies[i] = number of direct dependencies of expr with slot idx i
    num_dependencies: np.ndarray  # of int

    # records the output_expr that a subexpr belongs to
    # (a subexpr can be shared across multiple output exprs)
    output_expr_ids: list[set[int]]

    @dataclass
    class EvalCtx:
        """Context for evaluating a set of target exprs"""

        slot_idxs: list[int]  # slot idxs of exprs needed to evaluate target exprs; does not contain duplicates
        exprs: list[Expr]  # exprs corresponding to slot_idxs
        target_slot_idxs: list[int]  # slot idxs of target exprs; might contain duplicates
        target_exprs: list[Expr]  # exprs corresponding to target_slot_idxs

    def __init__(
        self,
        output_exprs: Sequence[Expr],
        columns: Sequence[catalog.Column],
        input_exprs: Iterable[Expr],
        tbl: Optional[catalog.TableVersion] = None,
    ):
        """
        Args:
            output_exprs: list of Exprs to be evaluated
            columns: list of columns to be materialized
            input_exprs: list of Exprs that are excluded from evaluation (because they're already materialized)
        TODO: enforce that output_exprs doesn't overlap with input_exprs?
        """
        self.unique_exprs: ExprSet[Expr] = ExprSet()  # dependencies precede their dependents
        self.next_slot_idx = 0

        # record input and output exprs; make copies to avoid reusing execution state
        unique_input_exprs = [self._record_unique_expr(e.copy(), recursive=False) for e in input_exprs]
        self.input_expr_slot_idxs = {e.slot_idx for e in unique_input_exprs}

        resolve_cols = set(columns)
        self.output_exprs = ExprSet(
            [
                self._record_unique_expr(e.copy().resolve_computed_cols(resolve_cols=resolve_cols), recursive=True)
                for e in output_exprs
            ]
        )

        # if init(columns):
        # - we are creating table rows and need to record columns for create_table_row()
        # - output_exprs materialize those columns
        # - input_exprs are ColumnRefs of the non-computed columns (ie, what needs to be provided as input)
        # - media validation:
        #   * for write-validated columns, we need to create validating ColumnRefs
        #   * further references to that column (eg, computed cols) need to resolve to the validating ColumnRef
        from .column_ref import ColumnRef

        self.tbl = tbl
        self.table_columns: list[ColumnSlotIdx] = []
        self.input_exprs = ExprSet()
        validating_colrefs: dict[Expr, Expr] = {}  # key: non-validating colref, value: corresp. validating colref
        for col in columns:
            expr: Expr
            if col.is_computed:
                assert col.value_expr is not None
                # create a copy here so we don't reuse execution state and resolve references to computed columns
                expr = col.value_expr.copy().resolve_computed_cols(resolve_cols=resolve_cols)
                expr = expr.substitute(validating_colrefs)
                expr = self._record_unique_expr(expr, recursive=True)
            else:
                # record a ColumnRef so that references to this column resolve to the same slot idx
                perform_validation = (
                    None
                    if not col.col_type.is_media_type()
                    else col.media_validation == catalog.MediaValidation.ON_WRITE
                )
                expr = ColumnRef(col, perform_validation=perform_validation)
                # recursive=True: needed for validating ColumnRef
                expr = self._record_unique_expr(expr, recursive=True)

                if perform_validation:
                    # if expr is a validating ColumnRef, the input is the non-validating ColumnRef
                    non_validating_colref = expr.components[0]
                    self.input_exprs.add(non_validating_colref)
                    validating_colrefs[non_validating_colref] = expr
                else:
                    self.input_exprs.add(expr)

            self.add_table_column(col, expr.slot_idx)
            self.output_exprs.add(expr)

        # default eval ctx: all output exprs
        self.default_eval_ctx = self.create_eval_ctx(list(self.output_exprs), exclude=unique_input_exprs)

        # references to unstored iterator columns:
        # - those ColumnRefs need to instantiate iterators
        # - we create and record the iterator args here and pass them to their respective ColumnRefs
        # - we do this instead of simply recording the iterator args as a component of those ColumnRefs,
        #   because that would cause them to be evaluated for every single row
        # - the separate eval ctx allows the ColumnRef to materialize the iterator args only when the underlying
        #   iterated object changes
        col_refs = [e for e in self.unique_exprs if isinstance(e, ColumnRef)]

        def refs_unstored_iter_col(col_ref: ColumnRef) -> bool:
            tbl = col_ref.col.tbl
            return tbl.is_component_view and tbl.is_iterator_column(col_ref.col) and not col_ref.col.is_stored

        unstored_iter_col_refs = [col_ref for col_ref in col_refs if refs_unstored_iter_col(col_ref)]
        component_views = [col_ref.col.tbl for col_ref in unstored_iter_col_refs]
        unstored_iter_args = {view.id: view.iterator_args.copy() for view in component_views}
        self.unstored_iter_args = {
            id: self._record_unique_expr(arg, recursive=True) for id, arg in unstored_iter_args.items()
        }

        for col_ref in unstored_iter_col_refs:
            iter_arg_ctx = self.create_eval_ctx([unstored_iter_args[col_ref.col.tbl.id]])
            col_ref.set_iter_arg_ctx(iter_arg_ctx)

        # we guarantee that we can compute the expr DAG in a single front-to-back pass
        for i, expr in enumerate(self.unique_exprs):
            assert expr.slot_idx == i

        # determine transitive dependencies for the purpose of exception propagation
        # (list of set of slot_idxs, indexed by slot_idx)
        # self.dependents = np.zeros((self.num_materialized, self.num_materialized), dtype=bool)
        self.dependencies = np.zeros((self.num_materialized, self.num_materialized), dtype=bool)
        exc_dependencies: list[set[int]] = [set() for _ in range(self.num_materialized)]
        from .column_property_ref import ColumnPropertyRef

        for expr in self.unique_exprs:
            if expr.slot_idx in self.input_expr_slot_idxs:
                # this is input and therefore doesn't depend on other exprs
                continue
            # error properties don't have exceptions themselves
            if isinstance(expr, ColumnPropertyRef) and expr.is_cellmd_prop():
                continue
            dependency_idxs = [d.slot_idx for d in expr.dependencies()]
            self.dependencies[expr.slot_idx, dependency_idxs] = True
            for d in expr.dependencies():
                exc_dependencies[expr.slot_idx].add(d.slot_idx)
                exc_dependencies[expr.slot_idx].update(exc_dependencies[d.slot_idx])

        self.num_dependencies = np.sum(self.dependencies, axis=1)
        self.dependents = self.dependencies.T
        self.transitive_dependents = np.zeros((self.num_materialized, self.num_materialized), dtype=bool)
        for i in reversed(range(self.num_materialized)):
            self.transitive_dependents[i] = self.dependents[i] | np.any(
                self.transitive_dependents[self.dependents[i]], axis=0
            )

        self._exc_dependents = [set() for _ in range(self.num_materialized)]
        for expr in self.unique_exprs:
            assert expr.slot_idx is not None
            for d_idx in exc_dependencies[expr.slot_idx]:
                self._exc_dependents[d_idx].add(expr.slot_idx)

        self.output_expr_ids = [set() for _ in range(self.num_materialized)]
        for e in self.output_exprs:
            self._record_output_expr_id(e, e.slot_idx)

    def add_table_column(self, col: catalog.Column, slot_idx: int) -> None:
        """Record a column that is part of the table row"""
        assert self.tbl is not None
        self.table_columns.append(ColumnSlotIdx(col, slot_idx))

    def output_slot_idxs(self) -> list[ColumnSlotIdx]:
        """Return ColumnSlotIdx for output columns"""
        return self.table_columns

    @property
    def num_materialized(self) -> int:
        return self.next_slot_idx

    def get_output_exprs(self) -> list[Expr]:
        """Returns exprs that were requested in the c'tor and require evaluation"""
        return list(self.output_exprs)

    def _next_slot_idx(self) -> int:
        result = self.next_slot_idx
        self.next_slot_idx += 1
        return result

    def _record_unique_expr(self, expr: Expr, recursive: bool) -> Expr:
        """Records the expr if it's not a duplicate and assigns a slot idx to expr and its components"
        Returns:
            the unique expr
        """
        if expr in self.unique_exprs:
            # expr is a duplicate: we use the original instead
            return self.unique_exprs[expr]

        # expr value needs to be computed via Expr.eval()
        if recursive:
            for i, c in enumerate(expr.components):
                # make sure we only refer to components that have themselves been recorded
                expr.components[i] = self._record_unique_expr(c, True)
        assert expr.slot_idx is None
        expr.slot_idx = self._next_slot_idx()
        self.unique_exprs.add(expr)
        return expr

    def _record_output_expr_id(self, e: Expr, output_expr_id: int) -> None:
        assert e.slot_idx is not None
        assert output_expr_id is not None
        if e.slot_idx in self.input_expr_slot_idxs:
            return
        self.output_expr_ids[e.slot_idx].add(output_expr_id)
        for d in e.dependencies():
            self._record_output_expr_id(d, output_expr_id)

    def _compute_dependencies(
        self, target_slot_idxs: list[int], excluded_slot_idxs: list[int], target_scope: Optional[ExprScope] = None
    ) -> list[int]:
        """Compute exprs needed to materialize the given target slots, excluding 'excluded_slot_idxs'

        If target_scope != None, stops transitive dependency resolution when leaving target_scope (ie, includes
        immediate dependents that aren't in target_scope, but doesn't resolve those).
        """
        dependencies: list[set[int]] = [set() for _ in range(self.num_materialized)]  # indexed by slot_idx
        # doing this front-to-back ensures that we capture transitive dependencies
        max_target_slot_idx = max(target_slot_idxs)
        for expr in self.unique_exprs:
            if expr.slot_idx > max_target_slot_idx:
                # we're done
                break
            if expr.slot_idx in excluded_slot_idxs:
                continue
            if expr.slot_idx in self.input_expr_slot_idxs:
                # this is input and therefore doesn't depend on other exprs
                continue
            if target_scope is not None and expr.scope() != target_scope:
                # don't resolve dependencies outside of target_scope
                continue
            for d in expr.dependencies():
                assert d.slot_idx is not None, f'{expr}, {d}'
                if d.slot_idx in excluded_slot_idxs:
                    continue
                dependencies[expr.slot_idx].add(d.slot_idx)
                dependencies[expr.slot_idx].update(dependencies[d.slot_idx])
        # merge dependencies and convert to list
        return sorted(set().union(*[dependencies[i] for i in target_slot_idxs]))

    def set_slot_idxs(self, expr_list: Sequence[Expr], remove_duplicates: bool = True) -> None:
        """
        Recursively sets slot_idx in expr_list and its components

        remove_duplicates == True: removes duplicates in-place
        """
        for e in expr_list:
            self.__set_slot_idxs_aux(e)
        if remove_duplicates:
            # only allowed if `expr_list` is a mutable list
            assert isinstance(expr_list, list)
            deduped = list(ExprSet(expr_list))
            expr_list[:] = deduped

    def __set_slot_idxs_aux(self, e: Expr) -> None:
        """Recursively sets slot_idx in e and its components"""
        if e not in self.unique_exprs:
            return
        e.slot_idx = self.unique_exprs[e].slot_idx
        for c in e.components:
            self.__set_slot_idxs_aux(c)

    def get_dependencies(
        self, targets: Iterable[Expr], exclude: Optional[Iterable[Expr]] = None, limit_scope: bool = True
    ) -> list[Expr]:
        """
        Return list of dependencies needed to evaluate the given target exprs (expressed as slot idxs).
        The exprs given in 'exclude' are excluded.
        If limit_scope == True, only returns dependencies in the same scope and immediate (ie, not transitive)
        dependencies from enclosing scopes.

        Returns:
            list of Exprs from unique_exprs (= with slot_idx set)
        """
        targets = list(targets)
        if exclude is None:
            exclude = []
        if len(targets) == 0:
            return []
        # make sure we only refer to recorded exprs
        targets = [self.unique_exprs[e] for e in targets]
        target_scope: Optional[ExprScope] = None
        if limit_scope:
            # make sure all targets are from the same scope
            target_scopes = {e.scope() for e in targets}
            assert len(target_scopes) == 1
            target_scope = target_scopes.pop()
        exclude = [self.unique_exprs[e] for e in exclude]
        target_slot_idxs = [e.slot_idx for e in targets]
        excluded_slot_idxs = [e.slot_idx for e in exclude]
        all_dependencies = set(
            self._compute_dependencies(target_slot_idxs, excluded_slot_idxs, target_scope=target_scope)
        )
        all_dependencies.update(target_slot_idxs)
        result_ids = list(all_dependencies)
        result_ids.sort()
        return [self.unique_exprs[id] for id in result_ids]

    def create_eval_ctx(
        self, targets: Iterable[Expr], exclude: Optional[Iterable[Expr]] = None, limit_scope: bool = True
    ) -> EvalCtx:
        """Return EvalCtx for targets"""
        targets = list(targets)
        if exclude is None:
            exclude = []
        if len(targets) == 0:
            return self.EvalCtx([], [], [], [])
        dependencies = self.get_dependencies(targets, exclude, limit_scope=limit_scope)
        targets = [self.unique_exprs[e] for e in targets]
        target_slot_idxs = [e.slot_idx for e in targets]
        ctx_slot_idxs = [e.slot_idx for e in dependencies]
        return self.EvalCtx(
            slot_idxs=ctx_slot_idxs,
            exprs=[self.unique_exprs[slot_idx] for slot_idx in ctx_slot_idxs],
            target_slot_idxs=target_slot_idxs,
            target_exprs=targets,
        )

    def set_exc(self, data_row: DataRow, slot_idx: int, exc: Exception) -> None:
        """Record an exception in data_row and propagate it to dependents"""
        data_row.set_exc(slot_idx, exc)
        for idx in self._exc_dependents[slot_idx]:
            data_row.set_exc(idx, exc)

    def eval(
        self,
        data_row: DataRow,
        ctx: EvalCtx,
        profile: Optional[ExecProfile] = None,
        ignore_errors: bool = False,
        force_eval: Optional[ExprScope] = None,
    ) -> None:
        """
        Populates the slots in data_row given in ctx.
        If an expr.eval() raises an exception, records the exception in the corresponding slot of data_row
        and omits any of that expr's dependents's eval().
        profile: if present, populated with execution time of each expr.eval() call; indexed by expr.slot_idx
        ignore_errors: if False, raises ExprEvalError if any expr.eval() raises an exception
        force_eval: forces exprs in the specified scope to be reevaluated, even if they already have a value
        """
        for expr in ctx.exprs:
            assert expr.slot_idx >= 0
            if expr.scope() != force_eval and (data_row.has_val[expr.slot_idx] or data_row.has_exc(expr.slot_idx)):
                continue
            try:
                start_time = time.perf_counter()
                expr.eval(data_row, self)
                if profile is not None:
                    profile.eval_time[expr.slot_idx] += time.perf_counter() - start_time
                    profile.eval_count[expr.slot_idx] += 1
            except Exception as exc:
                _, _, exc_tb = sys.exc_info()
                self.set_exc(data_row, expr.slot_idx, exc)
                if not ignore_errors:
                    input_vals = [data_row[d.slot_idx] for d in expr.dependencies()]
                    raise excs.ExprEvalError(
                        expr, f'expression {expr}', data_row.get_exc(expr.slot_idx), exc_tb, input_vals, 0
                    ) from exc

    def create_table_row(
        self, data_row: DataRow, cols_with_excs: Optional[set[int]], pk: tuple[int, ...]
    ) -> tuple[list[Any], int]:
        """Create a table row from the slots that have an output column assigned

        Return tuple[list of row values in `self.table_columns` order, # of exceptions]
            This excludes system columns.
        """
        from pixeltable.exprs.column_property_ref import ColumnPropertyRef

        num_excs = 0
        table_row: list[Any] = list(pk)
        for info in self.table_columns:
            col, slot_idx = info.col, info.slot_idx
            if data_row.has_exc(slot_idx):
                exc = data_row.get_exc(slot_idx)
                num_excs += 1
                if cols_with_excs is not None:
                    cols_with_excs.add(col.id)
                table_row.append(None)
                if col.stores_cellmd:
                    # exceptions get stored in the errortype/-msg properties of the cellmd column
                    table_row.append(ColumnPropertyRef.create_cellmd_exc(exc))
            else:
                if col.col_type.is_image_type() and data_row.file_urls[slot_idx] is None:
                    # we have yet to store this image
                    filepath = str(MediaStore.prepare_media_path(col.tbl.id, col.id, col.tbl.version))
                    data_row.flush_img(slot_idx, filepath)
                val = data_row.get_stored_val(slot_idx, col.get_sa_col_type())
                table_row.append(val)
                if col.stores_cellmd:
                    table_row.append(None)  # placeholder for cellmd column

        return table_row, num_excs

    def store_column_names(self) -> tuple[list[str], dict[int, catalog.Column]]:
        """
        Returns the list of store column names corresponding to the table_columns of this RowBuilder.
        The second tuple element of the return value is a dictionary containing all media columns in the
        table; it's the mapping {list_index: column}.
        """
        assert self.tbl is not None, self.table_columns
        store_col_names: list[str] = [pk_col.name for pk_col in self.tbl.store_tbl.pk_columns()]
        media_cols: dict[int, catalog.Column] = {}

        for col in self.table_columns:
            if col.col.col_type.is_media_type():
                media_cols[len(store_col_names)] = col.col
            store_col_names.append(col.col.store_name())
            if col.col.stores_cellmd:
                store_col_names.append(col.col.cellmd_store_name())

        return store_col_names, media_cols
