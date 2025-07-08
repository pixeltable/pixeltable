from __future__ import annotations

import dataclasses
import enum
from textwrap import dedent
from typing import Any, Iterable, Literal, Optional, Sequence
from uuid import UUID

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable import catalog, exceptions as excs, exec, exprs
from pixeltable.catalog import Column, TableVersionHandle
from pixeltable.exec.sql_node import OrderByClause, OrderByItem, combine_order_by_clauses, print_order_by_clause


def _is_agg_fn_call(e: exprs.Expr) -> bool:
    return isinstance(e, exprs.FunctionCall) and e.is_agg_fn_call and not e.is_window_fn_call


def _get_combined_ordering(
    o1: list[tuple[exprs.Expr, bool]], o2: list[tuple[exprs.Expr, bool]]
) -> list[tuple[exprs.Expr, bool]]:
    """Returns an ordering that's compatible with both o1 and o2, or an empty list if no such ordering exists"""
    result: list[tuple[exprs.Expr, bool]] = []
    # determine combined ordering
    for (e1, asc1), (e2, asc2) in zip(o1, o2):
        if e1.id != e2.id:
            return []
        if asc1 is not None and asc2 is not None and asc1 != asc2:
            return []
        asc = asc1 if asc1 is not None else asc2
        result.append((e1, asc))

    # add remaining ordering of the longer list
    prefix_len = min(len(o1), len(o2))
    if len(o1) > prefix_len:
        result.extend(o1[prefix_len:])
    elif len(o2) > prefix_len:
        result.extend(o2[prefix_len:])
    return result


class JoinType(enum.Enum):
    INNER = 0
    LEFT = 1
    # TODO: implement
    # RIGHT = 2
    FULL_OUTER = 3
    CROSS = 4

    LiteralType = Literal['inner', 'left', 'full_outer', 'cross']

    @classmethod
    def validated(cls, name: str, error_prefix: str) -> JoinType:
        try:
            return cls[name.upper()]
        except KeyError as exc:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.Error(f'{error_prefix} must be one of: [{val_strs}]') from exc


@dataclasses.dataclass
class JoinClause:
    """Corresponds to a single 'JOIN ... ON (...)' clause in a SELECT statement; excludes the joined table."""

    join_type: JoinType
    join_predicate: Optional[exprs.Expr]  # None for join_type == CROSS


@dataclasses.dataclass
class FromClause:
    """Corresponds to the From-clause ('FROM <tbl> JOIN ... ON (...) JOIN ...') of a SELECT statement"""

    tbls: list[catalog.TableVersionPath]
    join_clauses: list[JoinClause] = dataclasses.field(default_factory=list)

    @property
    def _first_tbl(self) -> catalog.TableVersionPath:
        assert len(self.tbls) == 1
        return self.tbls[0]


@dataclasses.dataclass
class SampleClause:
    """Defines a sampling clause for a table."""

    version: Optional[int]
    n: Optional[int]
    n_per_stratum: Optional[int]
    fraction: Optional[float]
    seed: Optional[int]
    stratify_exprs: Optional[list[exprs.Expr]]

    # This seed value is used if one is not supplied
    DEFAULT_SEED = 0

    # The version of the hashing algorithm used for ordering and fractional sampling.
    CURRENT_VERSION = 1

    def __post_init__(self) -> None:
        """If no version was provided, provide the default version"""
        if self.version is None:
            self.version = self.CURRENT_VERSION
        if self.seed is None:
            self.seed = self.DEFAULT_SEED

    @property
    def is_stratified(self) -> bool:
        """Check if the sampling is stratified"""
        return self.stratify_exprs is not None and len(self.stratify_exprs) > 0

    @property
    def is_repeatable(self) -> bool:
        """Return true if the same rows will continue to be sampled if source rows are added or deleted."""
        return not self.is_stratified and self.fraction is not None

    def display_str(self, inline: bool = False) -> str:
        return str(self)

    def as_dict(self) -> dict:
        """Return a dictionary representation of the object"""
        d = dataclasses.asdict(self)
        d['_classname'] = self.__class__.__name__
        if self.is_stratified:
            d['stratify_exprs'] = [e.as_dict() for e in self.stratify_exprs]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SampleClause:
        """Create a SampleClause from a dictionary representation"""
        d_cleaned = {key: value for key, value in d.items() if key != '_classname'}
        s = cls(**d_cleaned)
        if s.is_stratified:
            s.stratify_exprs = [exprs.Expr.from_dict(e) for e in d_cleaned.get('stratify_exprs', [])]
        return s

    def __repr__(self) -> str:
        s = ','.join(e.display_str(inline=True) for e in self.stratify_exprs)
        return (
            f'sample_{self.version}(n={self.n}, n_per_stratum={self.n_per_stratum}, '
            f'fraction={self.fraction}, seed={self.seed}, [{s}])'
        )

    @classmethod
    def fraction_to_md5_hex(cls, fraction: float) -> str:
        """Return the string representation of an approximation (to ~1e-9) of a fraction of the total space
        of md5 hash values.
        This is used for fractional sampling.
        """
        # Maximum count for the upper 32 bits of MD5: 2^32
        max_md5_value = (2**32) - 1

        # Calculate the fraction of this value
        threshold_int = max_md5_value * int(1_000_000_000 * fraction) // 1_000_000_000

        # Convert to hexadecimal string with padding
        return format(threshold_int, '08x') + 'ffffffffffffffffffffffff'


class Analyzer:
    """
    Performs semantic analysis of a query and stores the analysis state.
    """

    from_clause: FromClause
    all_exprs: list[exprs.Expr]  # union of all exprs, aside from sql_where_clause
    select_list: list[exprs.Expr]
    group_by_clause: Optional[list[exprs.Expr]]  # None for non-aggregate queries; [] for agg query w/o grouping
    grouping_exprs: list[exprs.Expr]  # [] for non-aggregate queries or agg query w/o grouping
    order_by_clause: OrderByClause
    stratify_exprs: list[exprs.Expr]  # [] if no stratiifcation is required
    sample_clause: Optional[SampleClause]  # None if no sampling clause is present

    sql_elements: exprs.SqlElementCache

    # Where clause of the Select stmt of the SQL scan
    sql_where_clause: Optional[exprs.Expr]

    # filter predicate applied to output rows of the SQL scan
    filter: Optional[exprs.Expr]

    agg_fn_calls: list[exprs.FunctionCall]  # grouping aggregation (ie, not window functions)
    window_fn_calls: list[exprs.FunctionCall]
    agg_order_by: list[exprs.Expr]

    def __init__(
        self,
        from_clause: FromClause,
        select_list: Sequence[exprs.Expr],
        where_clause: Optional[exprs.Expr] = None,
        group_by_clause: Optional[list[exprs.Expr]] = None,
        order_by_clause: Optional[list[tuple[exprs.Expr, bool]]] = None,
        sample_clause: Optional[SampleClause] = None,
    ):
        if order_by_clause is None:
            order_by_clause = []
        self.from_clause = from_clause
        self.sql_elements = exprs.SqlElementCache()

        # remove references to unstored computed cols
        self.select_list = [e.resolve_computed_cols() for e in select_list]
        if where_clause is not None:
            where_clause = where_clause.resolve_computed_cols()
        self.group_by_clause = (
            [e.resolve_computed_cols() for e in group_by_clause] if group_by_clause is not None else None
        )
        self.sample_clause = sample_clause
        if self.sample_clause is not None and self.sample_clause.is_stratified:
            self.stratify_exprs = [e.resolve_computed_cols() for e in sample_clause.stratify_exprs]
        else:
            self.stratify_exprs = []
        self.order_by_clause = [OrderByItem(e.resolve_computed_cols(), asc) for e, asc in order_by_clause]

        self.sql_where_clause = None
        self.filter = None
        if where_clause is not None:
            where_clause_conjuncts, self.filter = where_clause.split_conjuncts(self.sql_elements.contains)
            self.sql_where_clause = exprs.CompoundPredicate.make_conjunction(where_clause_conjuncts)

        # all exprs that are evaluated in Python; not executable
        self.all_exprs = self.select_list.copy()
        for join_clause in from_clause.join_clauses:
            if join_clause.join_predicate is not None:
                self.all_exprs.append(join_clause.join_predicate)
        if self.group_by_clause is not None:
            self.all_exprs.extend(self.group_by_clause)
        self.all_exprs.extend(self.stratify_exprs)
        self.all_exprs.extend(e for e, _ in self.order_by_clause)
        if self.filter is not None:
            if sample_clause is not None:
                raise excs.Error(f'Filter {self.filter} not expressible in SQL')
            self.all_exprs.append(self.filter)

        self.agg_order_by = []
        self.agg_fn_calls = []
        self.window_fn_calls = []
        self._analyze_agg()
        self.grouping_exprs = self.group_by_clause if self.group_by_clause is not None else []

    def _analyze_agg(self) -> None:
        """Check semantic correctness of aggregation and fill in agg-specific fields of Analyzer"""
        candidates = self.select_list
        agg_fn_calls = exprs.ExprSet(
            exprs.Expr.list_subexprs(
                candidates,
                expr_class=exprs.FunctionCall,
                filter=lambda e: bool(e.is_agg_fn_call and not e.is_window_fn_call),
            )
        )
        self.agg_fn_calls = list(agg_fn_calls)
        window_fn_calls = exprs.ExprSet(
            exprs.Expr.list_subexprs(
                candidates, expr_class=exprs.FunctionCall, filter=lambda e: bool(e.is_window_fn_call)
            )
        )
        self.window_fn_calls = list(window_fn_calls)
        if len(self.agg_fn_calls) == 0:
            # nothing to do
            return
        # if we're doing grouping aggregation and don't have an explicit Group By clause, we're creating a single group
        if self.group_by_clause is None:
            self.group_by_clause = []

        # check that select list only contains aggregate output
        grouping_expr_ids = {e.id for e in self.group_by_clause}
        is_agg_output = [self._determine_agg_status(e, grouping_expr_ids)[0] for e in self.select_list]
        if is_agg_output.count(False) > 0:
            raise excs.Error(
                f'Invalid non-aggregate expression in aggregate query: {self.select_list[is_agg_output.index(False)]}'
            )

        # check that Where clause and filter doesn't contain aggregates
        if self.sql_where_clause is not None and any(
            _is_agg_fn_call(e) for e in self.sql_where_clause.subexprs(expr_class=exprs.FunctionCall)
        ):
            raise excs.Error(f'where() cannot contain aggregate functions: {self.sql_where_clause}')
        if self.filter is not None and any(
            _is_agg_fn_call(e) for e in self.filter.subexprs(expr_class=exprs.FunctionCall)
        ):
            raise excs.Error(f'where() cannot contain aggregate functions: {self.filter}')

        # check that grouping exprs don't contain aggregates and can be expressed as SQL (we perform sort-based
        # aggregation and rely on the SqlScanNode returning data in the correct order)
        for e in self.group_by_clause:
            if not self.sql_elements.contains(e):
                raise excs.Error(f'Invalid grouping expression, needs to be expressible in SQL: {e}')
            if e._contains(filter=_is_agg_fn_call):
                raise excs.Error(f'Grouping expression contains aggregate function: {e}')

    def _determine_agg_status(self, e: exprs.Expr, grouping_expr_ids: set[int]) -> tuple[bool, bool]:
        """Determine whether expr is the input to or output of an aggregate function.
        Returns:
            (<is output>, <is input>)
        """
        if e.id in grouping_expr_ids:
            return True, True
        elif _is_agg_fn_call(e):
            for c in e.components:
                _, is_input = self._determine_agg_status(c, grouping_expr_ids)
                if not is_input:
                    raise excs.Error(f'Invalid nested aggregates: {e}')
            return True, False
        elif isinstance(e, exprs.Literal):
            return True, True
        elif isinstance(e, (exprs.ColumnRef, exprs.RowidRef)):
            # we already know that this isn't a grouping expr
            return False, True
        else:
            # an expression such as <grouping expr 1> + <grouping expr 2> can both be the output and input of agg
            assert len(e.components) > 0
            component_is_output, component_is_input = zip(
                *[self._determine_agg_status(c, grouping_expr_ids) for c in e.components]
            )
            is_output = component_is_output.count(True) == len(e.components)
            is_input = component_is_input.count(True) == len(e.components)
            if not is_output and not is_input:
                raise excs.Error(f'Invalid expression, mixes aggregate with non-aggregate: {e}')
            return is_output, is_input

    def finalize(self, row_builder: exprs.RowBuilder) -> None:
        """Make all exprs executable
        TODO: add EvalCtx for each expr list?
        """
        # maintain original composition of select list
        row_builder.set_slot_idxs(self.select_list, remove_duplicates=False)
        if self.group_by_clause is not None:
            row_builder.set_slot_idxs(self.group_by_clause)
        order_by_exprs = [e for e, _ in self.order_by_clause]
        row_builder.set_slot_idxs(order_by_exprs)
        row_builder.set_slot_idxs(self.all_exprs)
        if self.filter is not None:
            row_builder.set_slot_idxs([self.filter])
        row_builder.set_slot_idxs(self.agg_fn_calls)
        row_builder.set_slot_idxs(self.agg_order_by)

    def get_window_fn_ob_clause(self) -> Optional[OrderByClause]:
        clause: list[OrderByClause] = []
        for fn_call in self.window_fn_calls:
            # window functions require ordering by the group_by/order_by clauses
            group_by_exprs, order_by_exprs = fn_call.get_window_sort_exprs()
            clause.append(
                [OrderByItem(e, None) for e in group_by_exprs] + [OrderByItem(e, True) for e in order_by_exprs]
            )
        return combine_order_by_clauses(clause)

    def has_agg(self) -> bool:
        """True if there is any kind of aggregation in the query"""
        return self.group_by_clause is not None or len(self.agg_fn_calls) > 0 or len(self.window_fn_calls) > 0


class Planner:
    # TODO: create an exec.CountNode and change this to create_count_plan()
    @classmethod
    def create_count_stmt(cls, tbl: catalog.TableVersionPath, where_clause: Optional[exprs.Expr] = None) -> sql.Select:
        stmt = sql.select(sql.func.count().label('all_count'))
        refd_tbl_ids: set[UUID] = set()
        if where_clause is not None:
            analyzer = cls.analyze(tbl, where_clause)
            if analyzer.filter is not None:
                raise excs.Error(f'Filter {analyzer.filter} not expressible in SQL')
            clause_element = analyzer.sql_where_clause.sql_expr(analyzer.sql_elements)
            assert clause_element is not None
            stmt = stmt.where(clause_element)
            refd_tbl_ids = where_clause.tbl_ids()
        stmt = exec.SqlScanNode.create_from_clause(tbl, stmt, refd_tbl_ids)
        return stmt

    @classmethod
    def create_insert_plan(
        cls, tbl: catalog.TableVersion, rows: list[dict[str, Any]], ignore_errors: bool
    ) -> exec.ExecNode:
        """Creates a plan for TableVersion.insert()"""
        assert not tbl.is_view
        # stored_cols: all cols we need to store, incl computed cols (and indices)
        stored_cols = [c for c in tbl.cols_by_id.values() if c.is_stored]
        assert len(stored_cols) > 0  # there needs to be something to store

        cls.__check_valid_columns(tbl, stored_cols, 'inserted into')

        row_builder = exprs.RowBuilder([], stored_cols, [], tbl)

        # create InMemoryDataNode for 'rows'
        plan: exec.ExecNode = exec.InMemoryDataNode(
            TableVersionHandle(tbl.id, tbl.effective_version), rows, row_builder, tbl.next_row_id
        )

        media_input_col_info = [
            exprs.ColumnSlotIdx(col_ref.col, col_ref.slot_idx)
            for col_ref in row_builder.input_exprs
            if isinstance(col_ref, exprs.ColumnRef) and col_ref.col_type.is_media_type()
        ]
        if len(media_input_col_info) > 0:
            # prefetch external files for all input column refs
            plan = exec.CachePrefetchNode(tbl.id, media_input_col_info, input=plan)

        computed_exprs = row_builder.output_exprs - row_builder.input_exprs
        if len(computed_exprs) > 0:
            # add an ExprEvalNode when there are exprs to compute
            plan = exec.ExprEvalNode(
                row_builder, computed_exprs, plan.output_exprs, input=plan, maintain_input_order=False
            )

        stored_col_info = row_builder.output_slot_idxs()
        stored_img_col_info = [info for info in stored_col_info if info.col.col_type.is_image_type()]
        plan.set_stored_img_cols(stored_img_col_info)
        plan.set_ctx(
            exec.ExecContext(
                row_builder,
                batch_size=0,
                show_pbar=True,
                num_computed_exprs=len(computed_exprs),
                ignore_errors=ignore_errors,
            )
        )
        return plan

    @classmethod
    def rowid_columns(cls, target: TableVersionHandle, num_rowid_cols: Optional[int] = None) -> list[exprs.Expr]:
        """Return list of RowidRef for the given number of associated rowids"""
        if num_rowid_cols is None:
            num_rowid_cols = target.get().num_rowid_columns()
        return [exprs.RowidRef(target, i) for i in range(num_rowid_cols)]

    @classmethod
    def create_df_insert_plan(
        cls, tbl: catalog.TableVersion, df: 'pxt.DataFrame', ignore_errors: bool
    ) -> exec.ExecNode:
        assert not tbl.is_view
        plan = df._create_query_plan()  # ExecNode constructed by the DataFrame

        # Modify the plan RowBuilder to register the output columns
        for col_name, expr in zip(df.schema.keys(), df._select_list_exprs):
            assert col_name in tbl.cols_by_name
            col = tbl.cols_by_name[col_name]
            plan.row_builder.add_table_column(col, expr.slot_idx)

        stored_col_info = plan.row_builder.output_slot_idxs()
        stored_img_col_info = [info for info in stored_col_info if info.col.col_type.is_image_type()]
        plan.set_stored_img_cols(stored_img_col_info)

        plan.set_ctx(
            exec.ExecContext(
                plan.row_builder, batch_size=0, show_pbar=True, num_computed_exprs=0, ignore_errors=ignore_errors
            )
        )
        plan.ctx.num_rows = 0  # Unknown

        return plan

    @classmethod
    def create_update_plan(
        cls,
        tbl: catalog.TableVersionPath,
        update_targets: dict[catalog.Column, exprs.Expr],
        recompute_targets: list[catalog.Column],
        where_clause: Optional[exprs.Expr],
        cascade: bool,
    ) -> tuple[exec.ExecNode, list[str], list[catalog.Column]]:
        """Creates a plan to materialize updated rows.
        The plan:
        - retrieves rows that are visible at the current version of the table
        - materializes all stored columns and the update targets
        - if cascade is True, recomputes all computed columns that transitively depend on the updated columns
          and copies the values of all other stored columns
        - if cascade is False, copies all columns that aren't update targets from the original rows
        Returns:
            - root node of the plan
            - list of qualified column names that are getting updated
            - list of user-visible columns that are being recomputed
        """
        # retrieve all stored cols and all target exprs
        assert isinstance(tbl, catalog.TableVersionPath)
        target = tbl.tbl_version.get()  # the one we need to update
        updated_cols = list(update_targets.keys())
        recomputed_cols: set[Column]
        if len(recompute_targets) > 0:
            assert len(update_targets) == 0
            recomputed_cols = {*recompute_targets}
            if cascade:
                recomputed_cols |= target.get_dependent_columns(recomputed_cols)
        else:
            recomputed_cols = target.get_dependent_columns(updated_cols) if cascade else set()
        # regardless of cascade, we need to update all indices on any updated/recomputed column
        idx_val_cols = target.get_idx_val_columns(set(updated_cols) | recomputed_cols)
        recomputed_cols.update(idx_val_cols)
        # we only need to recompute stored columns (unstored ones are substituted away)
        recomputed_cols = {c for c in recomputed_cols if c.is_stored}

        cls.__check_valid_columns(tbl.tbl_version.get(), recomputed_cols, 'updated in')

        recomputed_base_cols = {col for col in recomputed_cols if col.tbl.id == tbl.tbl_version.id}
        copied_cols = [
            col
            for col in target.cols_by_id.values()
            if col.is_stored and col not in updated_cols and col not in recomputed_base_cols
        ]
        select_list: list[exprs.Expr] = [exprs.ColumnRef(col) for col in copied_cols]
        select_list.extend(update_targets.values())

        recomputed_exprs = [
            c.value_expr.copy().resolve_computed_cols(resolve_cols=recomputed_base_cols) for c in recomputed_base_cols
        ]
        # recomputed cols reference the new values of the updated cols
        spec: dict[exprs.Expr, exprs.Expr] = {exprs.ColumnRef(col): e for col, e in update_targets.items()}
        exprs.Expr.list_substitute(recomputed_exprs, spec)
        select_list.extend(recomputed_exprs)

        # we need to retrieve the PK columns of the existing rows
        plan = cls.create_query_plan(FromClause(tbls=[tbl]), select_list, where_clause=where_clause, ignore_errors=True)
        all_base_cols = copied_cols + updated_cols + list(recomputed_base_cols)  # same order as select_list
        # update row builder with column information
        for i, col in enumerate(all_base_cols):
            plan.row_builder.add_table_column(col, select_list[i].slot_idx)
        plan.ctx.num_computed_exprs = len(recomputed_exprs)
        recomputed_user_cols = [c for c in recomputed_cols if c.name is not None]
        return plan, [f'{c.tbl.name}.{c.name}' for c in updated_cols + recomputed_user_cols], recomputed_user_cols

    @classmethod
    def __check_valid_columns(
        cls, tbl: catalog.TableVersion, cols: Iterable[Column], op_name: Literal['inserted into', 'updated in']
    ) -> None:
        for col in cols:
            if col.value_expr is not None and not col.value_expr.is_valid:
                raise excs.Error(
                    dedent(
                        f"""
                        Data cannot be {op_name} the table {tbl.name!r},
                        because the column {col.name!r} is currently invalid:
                        {{validation_error}}
                        """
                    )
                    .strip()
                    .format(validation_error=col.value_expr.validation_error)
                )

    @classmethod
    def create_batch_update_plan(
        cls,
        tbl: catalog.TableVersionPath,
        batch: list[dict[catalog.Column, exprs.Expr]],
        rowids: list[tuple[int, ...]],
        cascade: bool,
    ) -> tuple[exec.ExecNode, exec.RowUpdateNode, sql.ColumnElement[bool], list[catalog.Column], list[catalog.Column]]:
        """
        Returns:
        - root node of the plan to produce the updated rows
        - RowUpdateNode of plan
        - Where clause for deleting the current versions of updated rows
        - list of columns that are getting updated
        - list of user-visible columns that are being recomputed
        """
        assert isinstance(tbl, catalog.TableVersionPath)
        target = tbl.tbl_version.get()  # the one we need to update
        sa_key_cols: list[sql.Column] = []
        key_vals: list[tuple] = []
        if len(rowids) > 0:
            sa_key_cols = target.store_tbl.rowid_columns()
            key_vals = rowids
        else:
            pk_cols = target.primary_key_columns()
            sa_key_cols = [c.sa_col for c in pk_cols]
            key_vals = [tuple(row[col].val for col in pk_cols) for row in batch]

        # retrieve all stored cols and all target exprs
        updated_cols = batch[0].keys() - target.primary_key_columns()
        recomputed_cols = target.get_dependent_columns(updated_cols) if cascade else set()
        # regardless of cascade, we need to update all indices on any updated column
        idx_val_cols = target.get_idx_val_columns(updated_cols)
        recomputed_cols.update(idx_val_cols)
        # we only need to recompute stored columns (unstored ones are substituted away)
        recomputed_cols = {c for c in recomputed_cols if c.is_stored}
        recomputed_base_cols = {col for col in recomputed_cols if col.tbl.id == target.id}
        copied_cols = [
            col
            for col in target.cols_by_id.values()
            if col.is_stored and col not in updated_cols and col not in recomputed_base_cols
        ]
        select_list: list[exprs.Expr] = [exprs.ColumnRef(col) for col in copied_cols]
        select_list.extend(exprs.ColumnRef(col) for col in updated_cols)

        recomputed_exprs = [
            c.value_expr.copy().resolve_computed_cols(resolve_cols=recomputed_base_cols) for c in recomputed_base_cols
        ]
        # the RowUpdateNode updates columns in-place, ie, in the original ColumnRef; no further substitution is needed
        select_list.extend(recomputed_exprs)

        # ExecNode tree (from bottom to top):
        # - SqlLookupNode to retrieve the existing rows
        # - RowUpdateNode to update the retrieved rows
        # - ExprEvalNode to evaluate the remaining output exprs
        analyzer = Analyzer(FromClause(tbls=[tbl]), select_list)
        sql_exprs = list(
            exprs.Expr.list_subexprs(analyzer.all_exprs, filter=analyzer.sql_elements.contains, traverse_matches=False)
        )
        row_builder = exprs.RowBuilder(analyzer.all_exprs, [], sql_exprs, target)
        analyzer.finalize(row_builder)
        sql_lookup_node = exec.SqlLookupNode(tbl, row_builder, sql_exprs, sa_key_cols, key_vals)
        col_vals = [{col: row[col].val for col in updated_cols} for row in batch]
        row_update_node = exec.RowUpdateNode(tbl, key_vals, len(rowids) > 0, col_vals, row_builder, sql_lookup_node)
        plan: exec.ExecNode = row_update_node
        if not cls._is_contained_in(analyzer.select_list, sql_exprs):
            # we need an ExprEvalNode to evaluate the remaining output exprs
            plan = exec.ExprEvalNode(row_builder, analyzer.select_list, sql_exprs, input=plan)
        # update row builder with column information
        all_base_cols = copied_cols + list(updated_cols) + list(recomputed_base_cols)  # same order as select_list
        row_builder.set_slot_idxs(select_list, remove_duplicates=False)
        for i, col in enumerate(all_base_cols):
            plan.row_builder.add_table_column(col, select_list[i].slot_idx)
        ctx = exec.ExecContext(row_builder, num_computed_exprs=len(recomputed_exprs))
        # we're returning everything to the user, so we might as well do it in a single batch
        ctx.batch_size = 0
        plan.set_ctx(ctx)
        recomputed_user_cols = [c for c in recomputed_cols if c.name is not None]
        return (
            plan,
            row_update_node,
            sql_lookup_node.where_clause_element,
            list(updated_cols) + recomputed_user_cols,
            recomputed_user_cols,
        )

    @classmethod
    def create_view_update_plan(
        cls, view: catalog.TableVersionPath, recompute_targets: list[catalog.Column]
    ) -> exec.ExecNode:
        """Creates a plan to materialize updated rows for a view, given that the base table has been updated.
        The plan:
        - retrieves rows that are visible at the current version of the table and satisfy the view predicate
        - materializes all stored columns and the update targets
        - if cascade is True, recomputes all computed columns that transitively depend on the updated columns
          and copies the values of all other stored columns
        - if cascade is False, copies all columns that aren't update targets from the original rows

        TODO: unify with create_view_load_plan()

        Returns:
            - root node of the plan
            - list of qualified column names that are getting updated
            - list of columns that are being recomputed
        """
        assert isinstance(view, catalog.TableVersionPath)
        assert view.is_view
        target = view.tbl_version.get()  # the one we need to update
        # retrieve all stored cols and all target exprs
        recomputed_cols = set(recompute_targets.copy())
        copied_cols = [col for col in target.cols_by_id.values() if col.is_stored and col not in recomputed_cols]
        select_list: list[exprs.Expr] = [exprs.ColumnRef(col) for col in copied_cols]
        # resolve recomputed exprs to stored columns in the base
        recomputed_exprs = [
            c.value_expr.copy().resolve_computed_cols(resolve_cols=recomputed_cols) for c in recomputed_cols
        ]
        select_list.extend(recomputed_exprs)

        # we need to retrieve the PK columns of the existing rows
        plan = cls.create_query_plan(
            FromClause(tbls=[view]),
            select_list,
            where_clause=target.predicate,
            ignore_errors=True,
            exact_version_only=view.get_bases(),
        )
        plan.ctx.num_computed_exprs = len(recomputed_exprs)
        for i, col in enumerate(copied_cols + list(recomputed_cols)):  # same order as select_list
            plan.row_builder.add_table_column(col, select_list[i].slot_idx)
        # TODO: avoid duplication with view_load_plan() logic (where does this belong?)
        stored_img_col_info = [
            info for info in plan.row_builder.output_slot_idxs() if info.col.col_type.is_image_type()
        ]
        plan.set_stored_img_cols(stored_img_col_info)
        return plan

    @classmethod
    def create_view_load_plan(
        cls, view: catalog.TableVersionPath, propagates_insert: bool = False
    ) -> tuple[exec.ExecNode, int]:
        """Creates a query plan for populating a view.

        Args:
            view: the view to populate
            propagates_insert: if True, we're propagating a base update to this view

        Returns:
            - root node of the plan
            - number of materialized values per row
        """
        assert isinstance(view, catalog.TableVersionPath)
        assert view.is_view
        # things we need to materialize as DataRows:
        # 1. stored computed cols
        # - iterator columns are effectively computed, just not with a value_expr
        # - we can ignore stored non-computed columns because they have a default value that is supplied directly by
        #   the store
        target = view.tbl_version.get()  # the one we need to populate
        stored_cols = [c for c in target.cols_by_id.values() if c.is_stored]
        # 2. for component views: iterator args
        iterator_args = [target.iterator_args] if target.iterator_args is not None else []

        from_clause = FromClause(tbls=[view.base])
        base_analyzer = Analyzer(
            from_clause, iterator_args, where_clause=target.predicate, sample_clause=target.sample_clause
        )
        row_builder = exprs.RowBuilder(base_analyzer.all_exprs, stored_cols, [], target)

        # if we're propagating an insert, we only want to see those base rows that were created for the current version
        # execution plan:
        # 1. materialize exprs computed from the base that are needed for stored view columns
        # 2. if it's an iterator view, expand the base rows into component rows
        # 3. materialize stored view columns that haven't been produced by step 1
        base_output_exprs = [e for e in row_builder.default_eval_ctx.exprs if e.is_bound_by([view.base])]
        view_output_exprs = [
            e
            for e in row_builder.default_eval_ctx.target_exprs
            if e.is_bound_by([view]) and not e.is_bound_by([view.base])
        ]

        # Create a new analyzer reflecting exactly what is required from the base table
        base_analyzer = Analyzer(
            from_clause, base_output_exprs, where_clause=target.predicate, sample_clause=target.sample_clause
        )
        base_eval_ctx = row_builder.create_eval_ctx(base_analyzer.all_exprs)
        plan = cls._create_query_plan(
            row_builder=row_builder,
            analyzer=base_analyzer,
            eval_ctx=base_eval_ctx,
            with_pk=True,
            exact_version_only=view.get_bases() if propagates_insert else [],
        )
        exec_ctx = plan.ctx
        if target.is_component_view:
            plan = exec.ComponentIterationNode(view.tbl_version, plan)
        if len(view_output_exprs) > 0:
            plan = exec.ExprEvalNode(
                row_builder, output_exprs=view_output_exprs, input_exprs=base_output_exprs, input=plan
            )

        stored_img_col_info = [info for info in row_builder.output_slot_idxs() if info.col.col_type.is_image_type()]
        plan.set_stored_img_cols(stored_img_col_info)
        exec_ctx.ignore_errors = True
        plan.set_ctx(exec_ctx)
        return plan, len(row_builder.default_eval_ctx.target_exprs)

    @classmethod
    def _verify_join_clauses(cls, analyzer: Analyzer) -> None:
        """Verify that join clauses are expressible in SQL"""
        for join_clause in analyzer.from_clause.join_clauses:
            if join_clause.join_predicate is not None and analyzer.sql_elements.get(join_clause.join_predicate) is None:
                raise excs.Error(f'Join predicate {join_clause.join_predicate} not expressible in SQL')

    @classmethod
    def _create_combined_ordering(cls, analyzer: Analyzer, verify_agg: bool) -> Optional[OrderByClause]:
        """Verify that the various ordering requirements don't conflict and return a combined ordering"""
        ob_clauses: list[OrderByClause] = [analyzer.order_by_clause.copy()]

        if verify_agg:
            ordering: OrderByClause
            for fn_call in analyzer.window_fn_calls:
                # window functions require ordering by the group_by/order_by clauses
                gb, ob = fn_call.get_window_sort_exprs()
                ordering = [OrderByItem(e, None) for e in gb] + [OrderByItem(e, True) for e in ob]
                ob_clauses.append(ordering)
            for fn_call in analyzer.agg_fn_calls:
                # agg functions with an ordering requirement are implicitly ascending
                ordering = [OrderByItem(e, None) for e in analyzer.group_by_clause] + [
                    OrderByItem(e, True) for e in fn_call.get_agg_order_by()
                ]
                ob_clauses.append(ordering)

        if len(ob_clauses) == 0:
            return None
        elif len(ob_clauses) == 1:
            return ob_clauses[0]

        combined_ordering = ob_clauses[0]
        for ordering in ob_clauses[1:]:
            combined = combine_order_by_clauses([combined_ordering, ordering])
            if combined is None:
                raise excs.Error(
                    f'Incompatible ordering requirements: '
                    f'{print_order_by_clause(combined_ordering)} vs {print_order_by_clause(ordering)}'
                )
            combined_ordering = combined
        return combined_ordering

    @classmethod
    def _is_contained_in(cls, l1: Iterable[exprs.Expr], l2: Iterable[exprs.Expr]) -> bool:
        """Returns True if l1 is contained in l2"""
        return {e.id for e in l1} <= {e.id for e in l2}

    @classmethod
    def _insert_prefetch_node(
        cls, tbl_id: UUID, row_builder: exprs.RowBuilder, input_node: exec.ExecNode
    ) -> exec.ExecNode:
        """Returns a CachePrefetchNode into the plan if needed, otherwise returns input"""
        # we prefetch external files for all media ColumnRefs, even those that aren't part of the dependencies
        # of output_exprs: if unstored iterator columns are present, we might need to materialize ColumnRefs that
        # aren't explicitly captured as dependencies
        media_col_refs = [
            e for e in list(row_builder.unique_exprs) if isinstance(e, exprs.ColumnRef) and e.col_type.is_media_type()
        ]
        if len(media_col_refs) == 0:
            return input_node
        # we need to prefetch external files for media column types
        file_col_info = [exprs.ColumnSlotIdx(e.col, e.slot_idx) for e in media_col_refs]
        prefetch_node = exec.CachePrefetchNode(tbl_id, file_col_info, input_node)
        return prefetch_node

    @classmethod
    def create_query_plan(
        cls,
        from_clause: FromClause,
        select_list: Optional[list[exprs.Expr]] = None,
        where_clause: Optional[exprs.Expr] = None,
        group_by_clause: Optional[list[exprs.Expr]] = None,
        order_by_clause: Optional[list[tuple[exprs.Expr, bool]]] = None,
        limit: Optional[exprs.Expr] = None,
        sample_clause: Optional[SampleClause] = None,
        ignore_errors: bool = False,
        exact_version_only: Optional[list[catalog.TableVersionHandle]] = None,
    ) -> exec.ExecNode:
        """Return plan for executing a query.
        Updates 'select_list' in place to make it executable.
        TODO: make exact_version_only a flag and use the versions from tbl
        """
        if select_list is None:
            select_list = []
        if order_by_clause is None:
            order_by_clause = []
        if exact_version_only is None:
            exact_version_only = []

        analyzer = Analyzer(
            from_clause,
            select_list,
            where_clause=where_clause,
            group_by_clause=group_by_clause,
            order_by_clause=order_by_clause,
            sample_clause=sample_clause,
        )
        # If the from_clause has a single table, we can use it as the context table for the RowBuilder.
        # Otherwise there is no context table, but that's ok, because the context table is only needed for
        # table mutations, which can't happen during a join.
        context_tbl = from_clause.tbls[0].tbl_version.get() if len(from_clause.tbls) == 1 else None
        row_builder = exprs.RowBuilder(analyzer.all_exprs, [], [], context_tbl)

        analyzer.finalize(row_builder)
        # select_list: we need to materialize everything that's been collected
        # with_pk: for now, we always retrieve the PK, because we need it for the file cache
        eval_ctx = row_builder.create_eval_ctx(analyzer.select_list)
        plan = cls._create_query_plan(
            row_builder=row_builder,
            analyzer=analyzer,
            eval_ctx=eval_ctx,
            limit=limit,
            with_pk=True,
            exact_version_only=exact_version_only,
        )
        plan.ctx.ignore_errors = ignore_errors
        select_list.clear()
        select_list.extend(analyzer.select_list)
        return plan

    @classmethod
    def _create_query_plan(
        cls,
        row_builder: exprs.RowBuilder,
        analyzer: Analyzer,
        eval_ctx: exprs.RowBuilder.EvalCtx,
        limit: Optional[exprs.Expr] = None,
        with_pk: bool = False,
        exact_version_only: Optional[list[catalog.TableVersionHandle]] = None,
    ) -> exec.ExecNode:
        """
        Create plan to materialize eval_ctx.

        Args:
            plan_target: if not None, generate a plan that materializes only expression that can be evaluted
                in the context of that table version (eg, if 'tbl' is a view, 'plan_target' might be the base)
        TODO: make exact_version_only a flag and use the versions from tbl
        """
        if exact_version_only is None:
            exact_version_only = []
        sql_elements = analyzer.sql_elements
        is_python_agg = not sql_elements.contains_all(analyzer.agg_fn_calls) or not sql_elements.contains_all(
            analyzer.window_fn_calls
        )
        ctx = exec.ExecContext(row_builder)
        combined_ordering = cls._create_combined_ordering(analyzer, verify_agg=is_python_agg)
        cls._verify_join_clauses(analyzer)

        # materialized with SQL table scans (ie, single-table SELECT statements):
        # - select list subexprs that aren't aggregates
        # - join clause subexprs
        # - subexprs of Where clause conjuncts that can't be run in SQL
        # - all grouping exprs
        # - all stratify exprs
        candidates = list(
            exprs.Expr.list_subexprs(
                analyzer.select_list,
                filter=lambda e: (
                    sql_elements.contains(e)
                    and not e._contains(cls=exprs.FunctionCall, filter=lambda e: bool(e.is_agg_fn_call))
                ),
                traverse_matches=False,
            )
        )
        if analyzer.filter is not None:
            candidates.extend(
                exprs.Expr.subexprs(analyzer.filter, filter=sql_elements.contains, traverse_matches=False)
            )
        candidates.extend(
            exprs.Expr.list_subexprs(analyzer.grouping_exprs, filter=sql_elements.contains, traverse_matches=False)
        )
        candidates.extend(
            exprs.Expr.list_subexprs(analyzer.stratify_exprs, filter=sql_elements.contains, traverse_matches=False)
        )
        # not isinstance(...): we don't want to materialize Literals via a Select
        sql_exprs = exprs.ExprSet(e for e in candidates if not isinstance(e, exprs.Literal))

        # create table scans; each scan produces subexprs of (sql_exprs + join clauses)
        join_exprs = exprs.ExprSet(
            join_clause.join_predicate
            for join_clause in analyzer.from_clause.join_clauses
            if join_clause.join_predicate is not None
        )
        scan_target_exprs = sql_exprs | join_exprs
        tbl_scan_plans: list[exec.SqlScanNode] = []
        plan: exec.ExecNode
        for tbl in analyzer.from_clause.tbls:
            # materialize all subexprs of scan_target_exprs that are bound by tbl
            tbl_scan_exprs = exprs.ExprSet(
                exprs.Expr.list_subexprs(
                    scan_target_exprs,
                    filter=lambda e: e.is_bound_by([tbl]) and not isinstance(e, exprs.Literal),
                    traverse_matches=False,
                )
            )
            plan = exec.SqlScanNode(
                tbl, row_builder, select_list=tbl_scan_exprs, set_pk=with_pk, exact_version_only=exact_version_only
            )
            tbl_scan_plans.append(plan)

        if len(analyzer.from_clause.join_clauses) > 0:
            plan = exec.SqlJoinNode(
                row_builder,
                inputs=tbl_scan_plans,
                join_clauses=analyzer.from_clause.join_clauses,
                select_list=sql_exprs,
            )
        else:
            plan = tbl_scan_plans[0]

        if analyzer.sql_where_clause is not None:
            plan.set_where(analyzer.sql_where_clause)
        if analyzer.filter is not None:
            plan.set_py_filter(analyzer.filter)
        if len(analyzer.window_fn_calls) > 0:
            # we need to order the input for window functions
            plan.set_order_by(analyzer.get_window_fn_ob_clause())

        if analyzer.sample_clause is not None:
            plan = exec.SqlSampleNode(
                row_builder,
                input=plan,
                select_list=tbl_scan_exprs,
                sample_clause=analyzer.sample_clause,
                stratify_exprs=analyzer.stratify_exprs,
            )

        plan = cls._insert_prefetch_node(tbl.tbl_version.id, row_builder, plan)

        if analyzer.group_by_clause is not None:
            # we're doing grouping aggregation; the input of the AggregateNode are the grouping exprs plus the
            # args of the agg fn calls
            agg_input = exprs.ExprSet(analyzer.grouping_exprs.copy())
            for fn_call in analyzer.agg_fn_calls:
                agg_input.update(fn_call.components)
            if not sql_exprs.issuperset(agg_input):
                # we need an ExprEvalNode
                plan = exec.ExprEvalNode(row_builder, agg_input, sql_exprs, input=plan)

            # batch size for aggregation input: this could be the entire table, so we need to divide it into
            # smaller batches; at the same time, we need to make the batches large enough to amortize the
            # function call overhead
            ctx.batch_size = 16

            # do aggregation in SQL if all agg exprs can be translated
            if (
                sql_elements.contains_all(analyzer.select_list)
                and sql_elements.contains_all(analyzer.grouping_exprs)
                and isinstance(plan, exec.SqlNode)
                and plan.to_cte() is not None
            ):
                plan = exec.SqlAggregationNode(
                    row_builder, input=plan, select_list=analyzer.select_list, group_by_items=analyzer.group_by_clause
                )
            else:
                input_sql_node = plan.get_node(exec.SqlNode)
                assert combined_ordering is not None
                input_sql_node.set_order_by(combined_ordering)
                plan = exec.AggregationNode(
                    tbl.tbl_version,
                    row_builder,
                    analyzer.group_by_clause,
                    analyzer.agg_fn_calls + analyzer.window_fn_calls,
                    agg_input,
                    input=plan,
                )
                typecheck_dummy = analyzer.grouping_exprs + analyzer.agg_fn_calls + analyzer.window_fn_calls
                agg_output = exprs.ExprSet(typecheck_dummy)
                if not agg_output.issuperset(exprs.ExprSet(eval_ctx.target_exprs)):
                    # we need an ExprEvalNode to evaluate the remaining output exprs
                    plan = exec.ExprEvalNode(row_builder, eval_ctx.target_exprs, agg_output, input=plan)
        else:
            if not exprs.ExprSet(sql_exprs).issuperset(exprs.ExprSet(eval_ctx.target_exprs)):
                # we need an ExprEvalNode to evaluate the remaining output exprs
                plan = exec.ExprEvalNode(row_builder, eval_ctx.target_exprs, sql_exprs, input=plan)
            # we're returning everything to the user, so we might as well do it in a single batch
            # TODO: return smaller batches in order to increase inter-ExecNode parallelism
            ctx.batch_size = 0

        sql_node = plan.get_node(exec.SqlNode)
        if len(analyzer.order_by_clause) > 0:
            # we have the last SqlNode we created produce the ordering
            assert sql_node is not None
            sql_node.set_order_by(analyzer.order_by_clause)

        # if we don't need an ordered result, tell the ExprEvalNode not to maintain input order (which allows us to
        # return batches earlier)
        if sql_node is not None and len(sql_node.order_by_clause) == 0:
            expr_eval_node = plan.get_node(exec.ExprEvalNode)
            if expr_eval_node is not None:
                expr_eval_node.set_input_order(False)

        if limit is not None:
            assert isinstance(limit, exprs.Literal)
            plan.set_limit(limit.val)

        plan.set_ctx(ctx)
        return plan

    @classmethod
    def analyze(cls, tbl: catalog.TableVersionPath, where_clause: exprs.Expr) -> Analyzer:
        return Analyzer(FromClause(tbls=[tbl]), [], where_clause=where_clause)

    @classmethod
    def create_add_column_plan(cls, tbl: catalog.TableVersionPath, col: catalog.Column) -> exec.ExecNode:
        """Creates a plan for InsertableTable.add_column()
        Returns:
            plan: the plan to execute
            value_expr slot idx for the plan output (for computed cols)
        """
        assert isinstance(tbl, catalog.TableVersionPath)
        row_builder = exprs.RowBuilder(output_exprs=[], columns=[col], input_exprs=[], tbl=tbl.tbl_version.get())
        analyzer = Analyzer(FromClause(tbls=[tbl]), row_builder.default_eval_ctx.target_exprs)
        plan = cls._create_query_plan(
            row_builder=row_builder, analyzer=analyzer, eval_ctx=row_builder.default_eval_ctx, with_pk=True
        )
        plan.ctx.batch_size = 16
        plan.ctx.show_pbar = True
        plan.ctx.ignore_errors = True
        computed_exprs = row_builder.output_exprs - row_builder.input_exprs
        plan.ctx.num_computed_exprs = len(computed_exprs)  # we are adding a computed column, so we need to evaluate it

        # we want to flush images
        if col.is_computed and col.is_stored and col.col_type.is_image_type():
            plan.set_stored_img_cols(row_builder.output_slot_idxs())
        return plan
