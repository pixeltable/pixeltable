from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple, Set, Iterable
from dataclasses import dataclass
import time
import sys

from .expr import Expr
from .expr_set import ExprSet
from .data_row import DataRow
import pixeltable.utils as utils
import pixeltable.func as func
import pixeltable.exceptions as excs
import pixeltable.catalog as catalog


class ExecProfile:
    def __init__(self, row_builder: RowBuilder):
        self.eval_time = [0.0] * row_builder.num_materialized
        self.eval_count = [0] * row_builder.num_materialized
        self.row_builder = row_builder

    def print(self, num_rows: int) -> str:
        for i in range(self.row_builder.num_materialized):
            if self.eval_count[i] == 0:
                continue
            per_call_time = self.eval_time[i] / self.eval_count[i]
            calls_per_row = self.eval_count[i] / num_rows
            multiple_str = f'({calls_per_row}x)' if calls_per_row > 1 else ''
            print(f'{self.row_builder.unique_exprs[i]}: {utils.print_perf_counter_delta(per_call_time)} {multiple_str}')


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

    @dataclass
    class EvalCtx:
        """Context for evaluating a set of target exprs"""
        slot_idxs: List[int]  # slot idxs of exprs needed to evaluate target exprs; does not contain duplicates
        exprs: List[Expr]  # exprs corresponding to slot_idxs
        target_slot_idxs: List[int]  # slot idxs of target exprs; might contain duplicates
        target_exprs: List[Expr]  # exprs corresponding to target_slot_idxs

    def __init__(
            self, output_exprs: List[Expr], columns: List[catalog.Column], input_exprs: List[Expr]
    ):
        """
        Args:
            output_exprs: list of Exprs to be evaluated
            columns: list of columns to be materialized
        """
        self.unique_exprs = ExprSet()  # dependencies precede their dependents
        self.next_slot_idx = 0

        # record input and output exprs; make copies to avoid reusing execution state
        unique_input_exprs = [self._record_unique_expr(e.copy(), recursive=False) for e in input_exprs]
        self.input_expr_slot_idxs = {e.slot_idx for e in unique_input_exprs}

        # output exprs: all exprs the caller wants to materialize
        # - explicitly requested output_exprs
        # - values for computed columns
        resolve_cols = set(columns)
        self.output_exprs = [
            self._record_unique_expr(e.copy().resolve_computed_cols(resolve_cols=resolve_cols), recursive=True)
            for e in output_exprs
        ]

        # record columns for create_table_row()
        from .column_ref import ColumnRef
        self.table_columns: List[ColumnSlotIdx] = []
        for col in columns:
            if col.is_computed:
                assert col.value_expr is not None
                # create a copy here so we don't reuse execution state and resolve references to computed columns
                expr = col.value_expr.copy().resolve_computed_cols(resolve_cols=resolve_cols)
                expr = self._record_unique_expr(expr, recursive=True)
                self.add_table_column(col, expr.slot_idx)
                self.output_exprs.append(expr)
            else:
                # record a ColumnRef so that references to this column resolve to the same slot idx
                ref = ColumnRef(col)
                ref = self._record_unique_expr(ref, recursive=False)
                self.add_table_column(col, ref.slot_idx)

        # default eval ctx: all output exprs
        self.default_eval_ctx = self.create_eval_ctx(self.output_exprs, exclude=unique_input_exprs)

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
            return tbl.is_component_view() and tbl.is_iterator_column(col_ref.col) and not col_ref.col.is_stored
        unstored_iter_col_refs = [col_ref for col_ref in col_refs if refs_unstored_iter_col(col_ref)]
        component_views = [col_ref.col.tbl for col_ref in unstored_iter_col_refs]
        unstored_iter_args = {view.id: view.iterator_args.copy() for view in component_views}
        self.unstored_iter_args = \
            {id: self._record_unique_expr(arg, recursive=True) for id, arg in unstored_iter_args.items()}

        for col_ref in unstored_iter_col_refs:
            iter_arg_ctx = self.create_eval_ctx([unstored_iter_args[col_ref.col.tbl.id]])
            col_ref.set_iter_arg_ctx(iter_arg_ctx)

        # we guarantee that we can compute the expr DAG in a single front-to-back pass
        for i, expr in enumerate(self.unique_exprs):
            assert expr.slot_idx == i

        # record transitive dependencies (list of set of slot_idxs, indexed by slot_idx)
        self.dependencies: List[Set[int]] = [set() for _ in range(self.num_materialized)]
        for expr in self.unique_exprs:
            if expr.slot_idx in self.input_expr_slot_idxs:
                # this is input and therefore doesn't depend on other exprs
                continue
            for d in expr.dependencies():
                self.dependencies[expr.slot_idx].add(d.slot_idx)
                self.dependencies[expr.slot_idx].update(self.dependencies[d.slot_idx])

        # derive transitive dependents
        self.dependents: List[Set[int]] = [set() for _ in range(self.num_materialized)]
        for expr in self.unique_exprs:
            for d in self.dependencies[expr.slot_idx]:
                self.dependents[d].add(expr.slot_idx)

        # records the output_expr that a subexpr belongs to
        # (a subexpr can be shared across multiple output exprs)
        self.output_expr_ids: List[Set[int]] = [set() for _ in range(self.num_materialized)]
        for e in self.output_exprs:
            self._record_output_expr_id(e, e.slot_idx)

    def add_table_column(self, col: catalog.Column, slot_idx: int) -> None:
        """Record a column that is part of the table row"""
        self.table_columns.append(ColumnSlotIdx(col, slot_idx))

    def output_slot_idxs(self) -> List[ColumnSlotIdx]:
        """Return ColumnSlotIdx for output columns"""
        return self.table_columns

    @property
    def num_materialized(self) -> int:
        return self.next_slot_idx

    def get_output_exprs(self) -> List[Expr]:
        """Returns exprs that were requested in the c'tor and require evaluation"""
        return self.output_exprs

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
        assert expr.slot_idx < 0
        expr.slot_idx = self._next_slot_idx()
        self.unique_exprs.append(expr)
        return expr

    def _record_output_expr_id(self, e: Expr, output_expr_id: int) -> None:
        self.output_expr_ids[e.slot_idx].add(output_expr_id)
        for d in e.dependencies():
            self._record_output_expr_id(d, output_expr_id)

    def _compute_dependencies(self, target_slot_idxs: List[int], excluded_slot_idxs: List[int]) -> List[int]:
        """Compute exprs needed to materialize the given target slots, excluding 'excluded_slot_idxs'"""
        dependencies = [set() for _ in range(self.num_materialized)]  # indexed by slot_idx
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
            for d in expr.dependencies():
                if d.slot_idx in excluded_slot_idxs:
                    continue
                dependencies[expr.slot_idx].add(d.slot_idx)
                dependencies[expr.slot_idx].update(dependencies[d.slot_idx])
        # merge dependencies and convert to list
        return sorted(set().union(*[dependencies[i] for i in target_slot_idxs]))

    def substitute_exprs(self, expr_list: List[Expr], remove_duplicates: bool = True) -> None:
        """Substitutes exprs with their executable counterparts from unique_exprs and optionally removes duplicates"""
        i = 0
        unique_ids: Set[i] = set()  # slot idxs within expr_list
        while i < len(expr_list):
            unique_expr = self.unique_exprs[expr_list[i]]
            if unique_expr.slot_idx in unique_ids and remove_duplicates:
                del expr_list[i]
            else:
                expr_list[i] = unique_expr
                unique_ids.add(unique_expr.slot_idx)
                i += 1

    def get_dependencies(self, targets: List[Expr], exclude: Optional[List[Expr]] = None) -> List[Expr]:
        """
        Return list of dependencies needed to evaluate the given target exprs (expressed as slot idxs).
        The exprs given in 'exclude' are excluded.
        Returns:
            list of Exprs from unique_exprs (= with slot_idx set)
        """
        if exclude is None:
            exclude = []
        if len(targets) == 0:
            return []
        # make sure we only refer to recorded exprs
        targets = [self.unique_exprs[e] for e in targets]
        exclude = [self.unique_exprs[e] for e in exclude]
        target_slot_idxs = [e.slot_idx for e in targets]
        excluded_slot_idxs = [e.slot_idx for e in exclude]
        all_dependencies = set(self._compute_dependencies(target_slot_idxs, excluded_slot_idxs))
        all_dependencies.update(target_slot_idxs)
        result_ids = list(all_dependencies)
        result_ids.sort()
        return [self.unique_exprs[id] for id in result_ids]

    def create_eval_ctx(self, targets: List[Expr], exclude: Optional[List[Expr]] = None) -> EvalCtx:
        """Return EvalCtx for targets"""
        if exclude is None:
            exclude = []
        if len(targets) == 0:
            return self.EvalCtx([], [], [], [])
        dependencies = self.get_dependencies(targets, exclude)
        targets = [self.unique_exprs[e] for e in targets]
        target_slot_idxs = [e.slot_idx for e in targets]
        ctx_slot_idxs = [e.slot_idx for e in dependencies]
        return self.EvalCtx(
            slot_idxs=ctx_slot_idxs, exprs=[self.unique_exprs[slot_idx] for slot_idx in ctx_slot_idxs],
            target_slot_idxs=target_slot_idxs, target_exprs=targets)

    def set_exc(self, data_row: DataRow, slot_idx: int, exc: Exception) -> None:
        """Record an exception in data_row and propagate it to dependents"""
        data_row.set_exc(slot_idx, exc)
        for slot_idx in self.dependents[slot_idx]:
            data_row.set_exc(slot_idx, exc)

    def eval(
            self, data_row: DataRow, ctx: EvalCtx, profile: Optional[ExecProfile] = None, ignore_errors: bool = False
    ) -> None:
        """
        Populates the slots in data_row given in ctx.
        If an expr.eval() raises an exception, records the exception in the corresponding slot of data_row
        and omits any of that expr's dependents's eval().
        profile: if present, populated with execution time of each expr.eval() call; indexed by expr.slot_idx
        ignore_errors: if False, raises ExprEvalError if any expr.eval() raises an exception
        """
        for expr in ctx.exprs:
            assert expr.slot_idx >= 0
            if data_row.has_val[expr.slot_idx] or data_row.has_exc(expr.slot_idx):
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
                        expr, f'expression {expr}', data_row.get_exc(expr.slot_idx), exc_tb, input_vals, 0)

    def create_table_row(self, data_row: DataRow, exc_col_ids: Set[int]) -> Tuple[Dict[str, Any], int]:
        """Create a table row from the slots that have an output column assigned

        Return Tuple[dict that represents a stored row (can be passed to sql.insert()), # of exceptions]
            This excludes system columns.
        """
        num_excs = 0
        table_row: Dict[str, Any] = {}
        for info in self.table_columns:
            col, slot_idx = info.col, info.slot_idx
            if data_row.has_exc(slot_idx):
                # exceptions get stored in the errortype/-msg columns
                exc = data_row.get_exc(slot_idx)
                num_excs += 1
                exc_col_ids.add(col.id)
                table_row[col.store_name()] = None
                table_row[col.errortype_store_name()] = type(exc).__name__
                table_row[col.errormsg_store_name()] = str(exc)
            else:
                val = data_row.get_stored_val(slot_idx, col.sa_col.type)
                table_row[col.store_name()] = val
                # we unfortunately need to set these, even if there are no errors
                table_row[col.errortype_store_name()] = None
                table_row[col.errormsg_store_name()] = None

        return table_row, num_excs

