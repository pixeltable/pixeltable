from typing import Tuple, Optional, List, Set, Any, Dict
from uuid import UUID

import sqlalchemy as sql

import pixeltable.exec as exec
import pixeltable.func as func
from pixeltable import catalog
from pixeltable import exceptions as excs
from pixeltable import exprs


def _is_agg_fn_call(e: exprs.Expr) -> bool:
    return isinstance(e, exprs.FunctionCall) and e.is_agg_fn_call and not e.is_window_fn_call

def _get_combined_ordering(
        o1: List[Tuple[exprs.Expr, bool]], o2: List[Tuple[exprs.Expr, bool]]
) -> List[Tuple[exprs.Expr, bool]]:
    """Returns an ordering that's compatible with both o1 and o2, or an empty list if no such ordering exists"""
    result: List[Tuple[exprs.Expr, bool]] = []
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

class Analyzer:
    """Class to perform semantic analysis of a query and to store the analysis state"""

    def __init__(
            self, tbl: catalog.TableVersionPath, select_list: List[exprs.Expr],
            where_clause: Optional[exprs.Predicate] = None, group_by_clause: Optional[List[exprs.Expr]] = None,
            order_by_clause: Optional[List[Tuple[exprs.Expr, bool]]] = None):
        if group_by_clause is None:
            group_by_clause = []
        if order_by_clause is None:
            order_by_clause = []
        self.tbl = tbl

        # remove references to unstored computed cols
        self.select_list = [e.resolve_computed_cols() for e in select_list]
        if where_clause is not None:
            where_clause = where_clause.resolve_computed_cols()
        self.group_by_clause = [e.resolve_computed_cols() for e in group_by_clause]
        self.order_by_clause = [(e.resolve_computed_cols(), asc) for e, asc in order_by_clause]

        # Where clause of the Select stmt of the SQL scan
        self.sql_where_clause: Optional[exprs.Expr] = None
        # filter predicate applied to output rows of the SQL scan
        self.filter: Optional[exprs.Predicate] = None
        # not executable
        self.similarity_clause: Optional[exprs.ImageSimilarityPredicate] = None
        if where_clause is not None:
            where_clause_conjuncts, self.filter = where_clause.split_conjuncts(lambda e: e.sql_expr() is not None)
            self.sql_where_clause = exprs.CompoundPredicate.make_conjunction(where_clause_conjuncts)
            if self.filter is not None:
                similarity_clauses, self.filter = self.filter.split_conjuncts(
                    lambda e: isinstance(e, exprs.ImageSimilarityPredicate))
                if len(similarity_clauses) > 1:
                    raise excs.Error(f'More than one nearest() not supported')
                if len(similarity_clauses) == 1:
                    if len(self.order_by_clause) > 0:
                        raise excs.Error((
                            f'nearest() returns results in order of proximity and cannot be used in conjunction with '
                            f'order_by()'))
                    self.similarity_clause = similarity_clauses[0]
                    img_col = self.similarity_clause.img_col_ref.col
                    indexed_col_ids = {info.col.id for info in tbl.tbl_version.idxs_by_name.values()}
                    if img_col.id not in indexed_col_ids:
                        raise excs.Error(f'nearest() not available for unindexed column {img_col.name}')

        # all exprs that are evaluated in Python; not executable
        self.all_exprs = self.select_list.copy()
        self.all_exprs.extend(self.group_by_clause)
        self.all_exprs.extend([e for e, _ in self.order_by_clause])
        if self.filter is not None:
            self.all_exprs.append(self.filter)
        self.sql_exprs = list(exprs.Expr.list_subexprs(
            self.all_exprs, filter=lambda e: e.sql_expr() is not None, traverse_matches=False))

        # sql_exprs: exprs that can be expressed via SQL and are retrieved directly from the store
        # (we don't want to materialize literals via SQL, so we remove them here)
        self.sql_exprs = [e for e in self.sql_exprs if not isinstance(e, exprs.Literal)]

        self.agg_fn_calls: List[exprs.FunctionCall] = []
        self.agg_order_by: List[exprs.Expr] = []
        self._analyze_agg()

    def _analyze_agg(self) -> None:
        """Check semantic correctness of aggregation and fill in agg-specific fields of Analyzer"""
        self.agg_fn_calls = [e for e in self.all_exprs if _is_agg_fn_call(e)]
        if len(self.agg_fn_calls) == 0:
            # nothing to do
            return

        # check that select list only contains aggregate output
        grouping_expr_ids = {e.id for e in self.group_by_clause}
        is_agg_output = [self._determine_agg_status(e, grouping_expr_ids)[0] for e in self.select_list]
        if is_agg_output.count(False) > 0:
            raise excs.Error(
                f'Invalid non-aggregate expression in aggregate query: {self.select_list[is_agg_output.index(False)]}')

        # check that filter doesn't contain aggregates
        if self.filter is not None:
            agg_fn_calls = [e for e in self.filter.subexprs(filter=lambda e: _is_agg_fn_call(e))]
            if len(agg_fn_calls) > 0:
                raise excs.Error(f'Filter cannot contain aggregate functions: {self.filter}')

        # check that grouping exprs don't contain aggregates and can be expressed as SQL (we perform sort-based
        # aggregation and rely on the SqlScanNode returning data in the correct order)
        for e in self.group_by_clause:
            if e.sql_expr() is None:
                raise excs.Error(f'Invalid grouping expression, needs to be expressible in SQL: {e}')
            if e.contains(filter=lambda e: _is_agg_fn_call(e)):
                raise excs.Error(f'Grouping expression contains aggregate function: {e}')

        # check that agg fn calls don't have contradicting ordering requirements
        order_by: List[exprs.Exprs] = []
        order_by_origin: Optional[exprs.Expr] = None  # the expr that determines the ordering
        for agg_fn_call in self.agg_fn_calls:
            fn_call_order_by = agg_fn_call.get_agg_order_by()
            if len(fn_call_order_by) == 0:
                continue
            if len(order_by) == 0:
                order_by = fn_call_order_by
                order_by_origin = agg_fn_call
            else:
                combined = _get_combined_ordering(
                    [(e, True) for e in order_by], [(e, True) for e in fn_call_order_by])
                if len(combined) == 0:
                    raise excs.Error((
                        f"Incompatible ordering requirements between expressions '{order_by_origin}' and "
                        f"'{agg_fn_call}':\n"
                        f"{exprs.Expr.print_list(order_by)} vs {exprs.Expr.print_list(fn_call_order_by)}"
                    ))
        self.agg_order_by = order_by

    def _determine_agg_status(self, e: exprs.Expr, grouping_expr_ids: Set[int]) -> Tuple[bool, bool]:
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
        elif isinstance(e, exprs.ColumnRef) or isinstance(e, exprs.RowidRef):
            # we already know that this isn't a grouping expr
            return False, True
        else:
            # an expression such as <grouping expr 1> + <grouping expr 2> can both be the output and input of agg
            assert len(e.components) > 0
            component_is_output, component_is_input = zip(
                *[self._determine_agg_status(c, grouping_expr_ids) for c in e.components])
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
        row_builder.substitute_exprs(self.select_list, remove_duplicates=False)
        row_builder.substitute_exprs(self.group_by_clause)
        order_by_exprs = [e for e, _ in self.order_by_clause]
        row_builder.substitute_exprs(order_by_exprs)
        self.order_by_clause = [(e, asc) for e, (_, asc) in zip(order_by_exprs, self.order_by_clause)]
        row_builder.substitute_exprs(self.all_exprs)
        row_builder.substitute_exprs(self.sql_exprs)
        if self.filter is not None:
            self.filter = row_builder.unique_exprs[self.filter]
        row_builder.substitute_exprs(self.agg_fn_calls)
        row_builder.substitute_exprs(self.agg_order_by)


class Planner:
    # TODO: create an exec.CountNode and change this to create_count_plan()
    @classmethod
    def create_count_stmt(
            cls, tbl: catalog.TableVersionPath, where_clause: Optional[exprs.Predicate] = None
    ) -> sql.Select:
        stmt = sql.select(sql.func.count('*'))
        refd_tbl_ids: Set[UUID] = set()
        if where_clause is not None:
            analyzer = cls.analyze(tbl, where_clause)
            if analyzer.similarity_clause is not None:
                raise excs.Error('nearest() cannot be used with count()')
            if analyzer.filter is not None:
                raise excs.Error(f'Filter {analyzer.filter} not expressible in SQL')
            clause_element = analyzer.sql_where_clause.sql_expr()
            assert clause_element is not None
            stmt = stmt.where(clause_element)
            refd_tbl_ids = where_clause.tbl_ids()
        stmt = exec.SqlScanNode.create_from_clause(tbl, stmt, refd_tbl_ids)
        return stmt

    @classmethod
    def create_insert_plan(
            cls, tbl: catalog.TableVersion, rows: List[Dict[str, Any]], ignore_errors: bool
    ) -> exec.ExecNode:
        """Creates a plan for TableVersion.insert()"""
        assert not tbl.is_view()
        # stored_cols: all cols we need to store, incl computed cols (and indices)
        stored_cols = [c for c in tbl.cols if c.is_stored]
        assert len(stored_cols) > 0

        row_builder = exprs.RowBuilder([], stored_cols, [])

        # create InMemoryDataNode for 'rows'
        stored_col_info = row_builder.output_slot_idxs()
        stored_img_col_info = [info for info in stored_col_info if info.col.col_type.is_image_type()]
        input_col_info = [info for info in stored_col_info if not info.col.is_computed]
        plan = exec.InMemoryDataNode(tbl, rows, row_builder, tbl.next_rowid)

        media_input_cols = [info for info in input_col_info if info.col.col_type.is_media_type()]

        # prefetch external files for all input column refs for validation
        plan = exec.CachePrefetchNode(tbl.id, media_input_cols, plan)
        plan = exec.MediaValidationNode(row_builder, media_input_cols, input=plan)

        computed_exprs = row_builder.default_eval_ctx.target_exprs
        if len(computed_exprs) > 0:
            # add an ExprEvalNode when there are exprs to compute
            plan = exec.ExprEvalNode(row_builder, computed_exprs, [], input=plan)

        plan.set_stored_img_cols(stored_img_col_info)
        plan.set_ctx(
            exec.ExecContext(
                row_builder, batch_size=0, show_pbar=True, num_computed_exprs=len(computed_exprs),
                ignore_errors=ignore_errors))
        return plan

    @classmethod
    def create_update_plan(
            cls, tbl: catalog.TableVersionPath,
            update_targets: dict[catalog.Column, exprs.Expr],
            recompute_targets: List[catalog.Column],
            where_clause: Optional[exprs.Predicate], cascade: bool
    ) -> Tuple[exec.ExecNode, List[str], List[catalog.Column]]:
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
            - list of columns that are being recomputed
        """
        # retrieve all stored cols and all target exprs
        assert isinstance(tbl, catalog.TableVersionPath)
        target = tbl.tbl_version  # the one we need to update
        updated_cols = list(update_targets.keys())
        if len(recompute_targets) > 0:
            recomputed_cols = recompute_targets.copy()
        else:
            recomputed_cols = target.get_dependent_columns(updated_cols) if cascade else {}
            # we only need to recompute stored columns (unstored ones are substituted away)
            recomputed_cols = {c for c in recomputed_cols if c.is_stored}
        recomputed_base_cols = {col for col in recomputed_cols if col.tbl == target}
        copied_cols = [
            col for col in target.cols if col.is_stored and not col in updated_cols and not col in recomputed_base_cols
        ]
        select_list = [exprs.ColumnRef(col) for col in copied_cols]
        select_list.extend(update_targets.values())

        recomputed_exprs = \
            [c.value_expr.copy().resolve_computed_cols(resolve_cols=recomputed_base_cols) for c in recomputed_base_cols]
        # recomputed cols reference the new values of the updated cols
        for col, e in update_targets.items():
            exprs.Expr.list_substitute(recomputed_exprs, exprs.ColumnRef(col), e)
        select_list.extend(recomputed_exprs)

        # we need to retrieve the PK columns of the existing rows
        plan = cls.create_query_plan(tbl, select_list, where_clause=where_clause, with_pk=True, ignore_errors=True)
        all_base_cols = copied_cols + updated_cols + list(recomputed_base_cols)  # same order as select_list
        # update row builder with column information
        [plan.row_builder.add_table_column(col, select_list[i].slot_idx) for i, col in enumerate(all_base_cols)]
        return plan, [f'{c.tbl.name}.{c.name}' for c in updated_cols + list(recomputed_cols)], list(recomputed_cols)

    @classmethod
    def create_view_update_plan(
            cls, view: catalog.TableVersionPath, recompute_targets: List[catalog.Column]
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
        assert view.is_view()
        target = view.tbl_version  # the one we need to update
        # retrieve all stored cols and all target exprs
        recomputed_cols = set(recompute_targets.copy())
        copied_cols = [col for col in target.cols if col.is_stored and not col in recomputed_cols]
        select_list = [exprs.ColumnRef(col) for col in copied_cols]
        # resolve recomputed exprs to stored columns in the base
        recomputed_exprs = \
            [c.value_expr.copy().resolve_computed_cols(resolve_cols=recomputed_cols) for c in recomputed_cols]
        select_list.extend(recomputed_exprs)

        # we need to retrieve the PK columns of the existing rows
        plan = cls.create_query_plan(
            view, select_list, where_clause=target.predicate, with_pk=True, ignore_errors=True,
            exact_version_only=view.get_bases())
        [
            plan.row_builder.add_table_column(col, select_list[i].slot_idx)
            for i, col in enumerate(copied_cols + list(recomputed_cols))  # same order as select_list
        ]
        # TODO: avoid duplication with view_load_plan() logic (where does this belong?)
        stored_img_col_info = \
            [info for info in plan.row_builder.output_slot_idxs() if info.col.col_type.is_image_type()]
        plan.set_stored_img_cols(stored_img_col_info)
        return plan

    @classmethod
    def create_view_load_plan(
            cls, view: catalog.TableVersionPath, propagates_insert: bool = False
    ) -> Tuple[exec.ExecNode, int]:
        """Creates a query plan for populating a view.

        Args:
            view: the view to populate
            propagates_insert: if True, we're propagating a base update to this view

        Returns:
            - root node of the plan
            - number of materialized values per row
        """
        assert isinstance(view, catalog.TableVersionPath)
        assert view.is_view()
        # things we need to materialize as DataRows:
        # 1. stored computed cols
        # - iterator columns are effectively computed, just not with a value_expr
        # - we can ignore stored non-computed columns because they have a default value that is supplied directly by
        #   the store
        target = view.tbl_version  # the one we need to populate
        stored_cols = [c for c in target.cols if c.is_stored and (c.is_computed or target.is_iterator_column(c))]
        # 2. for component views: iterator args
        iterator_args = [target.iterator_args] if target.iterator_args is not None else []

        row_builder = exprs.RowBuilder(iterator_args, stored_cols, [])

        # execution plan:
        # 1. materialize exprs computed from the base that are needed for stored view columns
        # 2. if it's an iterator view, expand the base rows into component rows
        # 3. materialize stored view columns that haven't been produced by step 1
        base_output_exprs = [e for e in row_builder.default_eval_ctx.exprs if e.is_bound_by(view.base)]
        view_output_exprs = [
            e for e in row_builder.default_eval_ctx.target_exprs
            if e.is_bound_by(view) and not e.is_bound_by(view.base)
        ]
        # if we're propagating an insert, we only want to see those base rows that were created for the current version
        base_analyzer = Analyzer(view, base_output_exprs, where_clause=target.predicate)
        plan = cls._create_query_plan(
            view.base, row_builder=row_builder, analyzer=base_analyzer, with_pk=True,
            exact_version_only=view.get_bases() if propagates_insert else [])
        exec_ctx = plan.ctx
        if target.is_component_view():
            plan = exec.ComponentIterationNode(target, plan)
        if len(view_output_exprs) > 0:
            plan = exec.ExprEvalNode(
                row_builder, output_exprs=view_output_exprs, input_exprs=base_output_exprs,input=plan)

        stored_img_col_info = [info for info in row_builder.output_slot_idxs() if info.col.col_type.is_image_type()]
        plan.set_stored_img_cols(stored_img_col_info)
        exec_ctx.ignore_errors = True
        plan.set_ctx(exec_ctx)
        return plan, len(row_builder.default_eval_ctx.target_exprs)

    @classmethod
    def _determine_ordering(cls, analyzer: Analyzer) -> List[Tuple[exprs.Expr, bool]]:
        """Returns the exprs for the ORDER BY clause of the SqlScanNode"""
        order_by_items: List[Tuple[exprs.Expr, Optional[bool]]] = []
        order_by_origin: Optional[exprs.Expr] = None  # the expr that determines the ordering


        # window functions require ordering by the group_by/order_by clauses
        window_fn_calls = [
            e for e in analyzer.all_exprs if isinstance(e, exprs.FunctionCall) and e.is_window_fn_call
        ]
        if len(window_fn_calls) > 0:
            for fn_call in window_fn_calls:
                gb, ob = fn_call.get_window_sort_exprs()
                # for now, the ordering is implicitly ascending
                fn_call_ordering = [(e, None) for e in gb] + [(e, True) for e in ob]
                if len(order_by_items) == 0:
                    order_by_items = fn_call_ordering
                    order_by_origin = fn_call
                else:
                    # check for compatibility
                    other_order_by_clauses = fn_call_ordering
                    combined = _get_combined_ordering(order_by_items, other_order_by_clauses)
                    if len(combined) == 0:
                        raise excs.Error((
                            f"Incompatible ordering requirements between expressions '{order_by_origin}' and "
                            f"'{fn_call}':\n"
                            f"{exprs.Expr.print_list(order_by_items)} vs {exprs.Expr.print_list(other_order_by_clauses)}"
                        ))
                    order_by_items = combined

        if len(analyzer.group_by_clause) > 0:
            agg_ordering = [(e, None) for e in analyzer.group_by_clause] + [(e, True) for e in analyzer.agg_order_by]
            if len(order_by_items) > 0:
                # check for compatibility
                combined = _get_combined_ordering(order_by_items, agg_ordering)
                if len(combined) == 0:
                    raise excs.Error((
                        f"Incompatible ordering requirements between expressions '{order_by_origin}' and "
                        f"grouping expressions:\n"
                        f"{exprs.Expr.print_list([e for e, _ in order_by_items])} vs "
                        f"{exprs.Expr.print_list([e for e, _ in agg_ordering])}"
                    ))
                order_by_items = combined
            else:
                order_by_items = agg_ordering

        if len(analyzer.order_by_clause) > 0:
            if len(order_by_items) > 0:
                # check for compatibility
                combined = _get_combined_ordering(order_by_items, analyzer.order_by_clause)
                if len(combined) == 0:
                    raise excs.Error((
                        f"Incompatible ordering requirements between expressions '{order_by_origin}' and "
                        f"order-by expressions:\n"
                        f"{exprs.Expr.print_list([e for e, _ in order_by_items])} vs "
                        f"{exprs.Expr.print_list([e for e, _ in analyzer.order_by_clause])}"
                    ))
                order_by_items = combined
            else:
                order_by_items = analyzer.order_by_clause

        # TODO: can this be unified with the same logic in RowBuilder
        def refs_unstored_iter_col(e: exprs.Expr) -> bool:
            if not isinstance(e, exprs.ColumnRef):
                return False
            tbl = e.col.tbl
            return tbl.is_component_view() and tbl.is_iterator_column(e.col) and not e.col.is_stored
        unstored_iter_col_refs = list(exprs.Expr.list_subexprs(analyzer.all_exprs, filter=refs_unstored_iter_col))
        if len(unstored_iter_col_refs) > 0 and len(order_by_items) == 0:
            # we don't already have a user-requested ordering and we access unstored iterator columns:
            # order by the primary key of the component view, which minimizes the number of iterator instantiations
            component_views = {e.col.tbl for e in unstored_iter_col_refs}
            # TODO: generalize this to multi-level iteration
            assert len(component_views) == 1
            component_view = list(component_views)[0]
            order_by_items = [
                (exprs.RowidRef(component_view, idx), None)
                for idx in range(len(component_view.store_tbl.rowid_columns()))
            ]
            order_by_origin = unstored_iter_col_refs[0]

        for e in [e for e, _ in order_by_items]:
            if e.sql_expr() is None:
                raise excs.Error(f'order_by element cannot be expressed in SQL: {e}')
        # we do ascending ordering by default, if not specified otherwise
        order_by_items = [(e, True) if asc is None else (e, asc) for e, asc in order_by_items]
        return order_by_items

    @classmethod
    def _is_contained_in(cls, l1: List[exprs.Expr], l2: List[exprs.Expr]) -> bool:
        """Returns True if l1 is contained in l2"""
        s1, s2 = set([e.id for e in l1]), set([e.id for e in l2])
        return s1 <= s2

    @classmethod
    def _insert_prefetch_node(
            cls, tbl_id: UUID, output_exprs: List[exprs.Expr], row_builder: exprs.RowBuilder, input: exec.ExecNode
    ) -> exec.ExecNode:
        """Returns a CachePrefetchNode into the plan if needed, otherwise returns input"""
        # we prefetch external files for all media ColumnRefs, even those that aren't part of the dependencies
        # of output_exprs: if unstored iterator columns are present, we might need to materialize ColumnRefs that
        # aren't explicitly captured as dependencies
        media_col_refs = [
            e for e in list(row_builder.unique_exprs) if isinstance(e, exprs.ColumnRef) and e.col_type.is_media_type()
        ]
        if len(media_col_refs) == 0:
            return input
        # we need to prefetch external files for media column types
        file_col_info = [exprs.ColumnSlotIdx(e.col, e.slot_idx) for e in media_col_refs]
        prefetch_node = exec.CachePrefetchNode(tbl_id, file_col_info, input)
        return prefetch_node

    @classmethod
    def create_query_plan(
            cls, tbl: catalog.TableVersionPath, select_list: Optional[List[exprs.Expr]] = None,
            where_clause: Optional[exprs.Predicate] = None, group_by_clause: Optional[List[exprs.Expr]] = None,
            order_by_clause: Optional[List[Tuple[exprs.Expr, bool]]] = None, limit: Optional[int] = None,
            with_pk: bool = False, ignore_errors: bool = False, exact_version_only: Optional[List[catalog.TableVersion]] = None
    ) -> exec.ExecNode:
        """Return plan for executing a query.
        Updates 'select_list' in place to make it executable.
        TODO: make exact_version_only a flag and use the versions from tbl
        """
        if select_list is None:
            select_list = []
        if group_by_clause is None:
            group_by_clause = []
        if order_by_clause is None:
            order_by_clause = []
        if exact_version_only is None:
            exact_version_only = []
        analyzer = Analyzer(
            tbl, select_list, where_clause=where_clause, group_by_clause=group_by_clause,
            order_by_clause=order_by_clause)
        row_builder = exprs.RowBuilder(analyzer.all_exprs, [], analyzer.sql_exprs)

        analyzer.finalize(row_builder)
        # select_list: we need to materialize everything that's been collected
        # with_pk: for now, we always retrieve the PK, because we need it for the file cache
        plan = cls._create_query_plan(
            tbl, row_builder, analyzer=analyzer, limit=limit, with_pk=True, exact_version_only=exact_version_only)
        plan.ctx.ignore_errors = ignore_errors
        select_list.clear()
        select_list.extend(analyzer.select_list)
        return plan

    @classmethod
    def _create_query_plan(
            cls, tbl: catalog.TableVersionPath, row_builder: exprs.RowBuilder, analyzer: Analyzer,
            limit: Optional[int] = None, with_pk: bool = False, exact_version_only: Optional[List[catalog.TableVersion]] = None
    ) -> exec.ExecNode:
        """
        Args:
            plan_target: if not None, generate a plan that materializes only expression that can be evaluted
                in the context of that table version (eg, if 'tbl' is a view, 'plan_target' might be the base)
        TODO: make exact_version_only a flag and use the versions from tbl
        """
        if exact_version_only is None:
            exact_version_only = []
        assert isinstance(tbl, catalog.TableVersionPath)
        is_agg_query = len(analyzer.group_by_clause) > 0 or len(analyzer.agg_fn_calls) > 0
        ctx = exec.ExecContext(row_builder)

        order_by_items = cls._determine_ordering(analyzer)
        sql_limit = 0 if is_agg_query else limit  # if we're aggregating, the limit applies to the agg output
        sql_select_list = analyzer.sql_exprs.copy()
        plan = exec.SqlScanNode(
            tbl, row_builder, select_list=sql_select_list, where_clause=analyzer.sql_where_clause,
            filter=analyzer.filter, similarity_clause=analyzer.similarity_clause, order_by_items=order_by_items,
            limit=sql_limit, set_pk=with_pk, exact_version_only=exact_version_only)
        plan = cls._insert_prefetch_node(tbl.tbl_version.id, analyzer.select_list, row_builder, plan)

        if len(analyzer.group_by_clause) > 0 or len(analyzer.agg_fn_calls) > 0:
            # we're doing aggregation; the input of the AggregateNode are the grouping exprs plus the
            # args of the agg fn calls
            agg_input = exprs.ExprSet(analyzer.group_by_clause.copy())
            for fn_call in analyzer.agg_fn_calls:
                agg_input.extend(fn_call.components)
            if not cls._is_contained_in(agg_input, analyzer.sql_exprs):
                # we need an ExprEvalNode
                plan = exec.ExprEvalNode(row_builder, agg_input, analyzer.sql_exprs, input=plan)

            # batch size for aggregation input: this could be the entire table, so we need to divide it into
            # smaller batches; at the same time, we need to make the batches large enough to amortize the
            # function call overhead
            # TODO: increase this if we have NOS calls in order to reduce the cost of switching models, but take
            # into account the amount of memory needed for intermediate images
            ctx.batch_size = 16

            plan = exec.AggregationNode(
                tbl.tbl_version, row_builder, analyzer.group_by_clause, analyzer.agg_fn_calls, agg_input, input=plan)
            agg_output = analyzer.group_by_clause + analyzer.agg_fn_calls
            if not cls._is_contained_in(analyzer.select_list, agg_output):
                # we need an ExprEvalNode to evaluate the remaining output exprs
                plan = exec.ExprEvalNode(
                    row_builder, analyzer.select_list, agg_output, input=plan)
        else:
            if not cls._is_contained_in(analyzer.select_list, analyzer.sql_exprs):
                # we need an ExprEvalNode to evaluate the remaining output exprs
                plan = exec.ExprEvalNode(row_builder, analyzer.select_list, analyzer.sql_exprs, input=plan)
            # we're returning everything to the user, so we might as well do it in a single batch
            ctx.batch_size = 0

        plan.set_ctx(ctx)
        return plan

    @classmethod
    def analyze(cls, tbl: catalog.TableVersionPath, where_clause: exprs.Predicate) -> Analyzer:
        return Analyzer(tbl, [], where_clause=where_clause)

    @classmethod
    def create_add_column_plan(
            cls, tbl: catalog.TableVersionPath, col: catalog.Column
    ) -> Tuple[exec.ExecNode, Optional[int]]:
        """Creates a plan for InsertableTable.add_column()
        Returns:
            plan: the plan to execute
            value_expr slot idx for the plan output (for computed cols)
        """
        assert isinstance(tbl, catalog.TableVersionPath)
        index_info: List[Tuple[catalog.Column, func.Function]] = []
        row_builder = exprs.RowBuilder(output_exprs=[], columns=[col], input_exprs=[])
        analyzer = Analyzer(tbl, row_builder.default_eval_ctx.target_exprs)
        plan = cls._create_query_plan(tbl, row_builder=row_builder, analyzer=analyzer, with_pk=True)
        plan.ctx.batch_size = 16
        plan.ctx.show_pbar = True
        plan.ctx.ignore_errors = True

        # we want to flush images
        if col.is_computed and col.is_stored and col.col_type.is_image_type():
            plan.set_stored_img_cols(row_builder.output_slot_idxs())
        value_expr_slot_idx = row_builder.output_slot_idxs()[0].slot_idx if col.is_computed else None
        return plan, value_expr_slot_idx
