from typing import Tuple, Optional, List, Set, Any
from uuid import UUID

import sqlalchemy as sql

from pixeltable import catalog
from pixeltable import exprs
from pixeltable.exec import \
    ColumnInfo, ExecContext, ExprEvalNode, InsertDataNode, SqlScanNode, ExecNode, AggregationNode, CachePrefetchNode
from pixeltable import exceptions as exc

class Planner:

    # TODO: create an exec.CountNode and change this to create_count_plan()
    @classmethod
    def create_count_stmt(
            cls, tbl: catalog.TableVersion, where_clause: Optional[exprs.Predicate] = None
    ) -> sql.Select:
        # identify the base table
        base_tbl = tbl
        while base_tbl.base is not None:
            base_tbl = base_tbl.base
        stmt = sql.select(sql.func.count('*')).select_from(base_tbl.store_tbl.sa_tbl)
        stmt = stmt \
            .where(base_tbl.store_tbl.v_min_col <= base_tbl.version) \
            .where(base_tbl.store_tbl.v_max_col > base_tbl.version)

        # join views
        # TODO: omit views that don't have filters
        while tbl.base is not None:
            assert tbl.is_view()
            # join with rows in tbl that are visible at the current version of base_tbl;
            # we can ignore the view predicate here, it's indirectly applied via the join
            stmt = stmt.join(tbl.store_tbl.sa_tbl, base_tbl.store_tbl.rowid_col == tbl.store_tbl.base_rowid_col) \
                .where(tbl.store_tbl.base_v_min_col <= base_tbl.version) \
                .where(tbl.store_tbl.base_v_max_col > base_tbl.version) \
                .where(tbl.store_tbl.v_min_col <= tbl.version) \
                .where(tbl.store_tbl.v_max_col > tbl.version)
            tbl = tbl.base

        if where_clause is not None:
            analysis_info = cls.get_info(tbl, where_clause)
            if analysis_info.similarity_clause is not None:
                raise exc.Error('nearest() cannot be used with count()')
            if analysis_info.filter is not None:
                raise exc.Error(f'Filter {analysis_info.filter} not expressible in SQL')
            stmt = stmt.where(analysis_info.sql_where_clause)
        return stmt

    @classmethod
    def create_insert_plan(
            cls, tbl: catalog.TableVersion, rows: List[List[Any]], column_names: List[str]
    ) -> Tuple[ExecNode, List[ColumnInfo], List[ColumnInfo], int]:
        """Creates a plan for Table.insert()

        Returns:
            - root node of the plan
            - info for cols stored in the db
            - info for cols stored in the NN index
            - number of materialized values per row
        """
        assert not tbl.is_view()
        # stored_cols: all cols we need to store, incl computed cols (and indices)
        stored_cols = [c for c in tbl.cols if c.is_stored and (c.name in column_names or c.is_computed)]
        if tbl.extracts_frames():
            stored_cols.append(tbl.frame_idx_col())

        # we include embeddings for indices by constructing computed cols; we need to do that for all indexed cols,
        # not just the stored ones
        indexed_cols = [c for c in tbl.cols if c.is_indexed]
        indexed_col_refs = \
            [exprs.FrameColumnRef(c) if tbl.is_frame_col(c) else exprs.ColumnRef(c) for c in indexed_cols]
        from pixeltable.functions.image_embedding import openai_clip
        # explicitly resize images to the required size
        target_img_type = next(iter(openai_clip.md.signature.parameters.values()))
        index_cols = [
            catalog.Column('dummy', computed_with=openai_clip(col_ref.resize(target_img_type.size)), stored=True)
            for col_ref in indexed_col_refs
        ]
        stored_cols.extend(index_cols)
        assert len(stored_cols) > 0

        # create copies to avoid reusing past execution state; eval ctx and evaluator need to share these copies;
        # also, we need to get rid of references to computed cols
        stored_exprs = [
            c.value_expr.copy().resolve_computed_cols(unstored_only=False) if c.is_computed else exprs.ColumnRef(c)
            for c in stored_cols
        ]
        evaluator = exprs.Evaluator(stored_exprs)
        num_db_cols = len(stored_cols) - len(index_cols)
        db_col_info = [ColumnInfo(stored_cols[i], stored_exprs[i].slot_idx) for i in range(num_db_cols)]
        idx_col_range = range(len(stored_cols) - len(index_cols), len(stored_cols))
        idx_col_info = \
            [ColumnInfo(indexed_cols[i - num_db_cols], stored_exprs[i].slot_idx) for i in idx_col_range]

        # create InsertDataNode for 'rows'
        stored_col_info = [ColumnInfo(c, e.slot_idx) for c, e in zip(stored_cols, stored_exprs)]
        stored_img_col_info = [info for info in stored_col_info if info.col.col_type.is_image_type()]
        frame_idx_col, frame_idx_slot_idx = tbl.frame_idx_col(), None
        if frame_idx_col is not None:
            frame_idx_slot_idx = stored_col_info[stored_cols.index(frame_idx_col)].slot_idx
        input_col_info = \
            [info for info in stored_col_info if not info.col.is_computed and not info.col == frame_idx_col]
        row_column_pos = {name: i for i, name in enumerate(column_names)}
        plan = InsertDataNode(tbl, rows, row_column_pos, evaluator, input_col_info, frame_idx_slot_idx, tbl.next_rowid)

        # add an ExprEvalNode if there are columns to compute
        computed_col_info = [c for c in stored_col_info if c.col.is_computed]
        if len(computed_col_info) > 0:
            uncomputed_col_info = [c for c in stored_col_info if not c.col.is_computed]
            computed_col_exprs = [evaluator.unique_exprs[i.slot_idx] for i in computed_col_info]
            # prefetch external files for media column types
            plan = cls._insert_prefetch_node(tbl.id, computed_col_exprs, evaluator, plan)
            plan = ExprEvalNode(
                evaluator, computed_col_exprs, [evaluator.unique_exprs[i.slot_idx] for i in uncomputed_col_info],
                ignore_errors=True, input=plan)
        plan.set_stored_img_cols(stored_img_col_info)
        plan.set_ctx(ExecContext(evaluator, batch_size=0, show_pbar=True))
        return plan, db_col_info, idx_col_info, len(computed_col_info)

    @classmethod
    def create_view_load_plan(
            cls, view: catalog.TableVersion, base_version: Optional[int] = None
    ) -> Tuple[ExecNode, List[ColumnInfo], List[ColumnInfo], int]:
        """Creates a query plan for populating a view.

        Args:
            view: the view to populate
            base_version: if set, only retrieves rows with version == base_version (not all visible rows)
        Returns:
            - root node of the plan
            - info for cols stored in the db
            - info for cols stored in the NN index
            - number of materialized values per row
        """
        assert view.is_view()
        # select list:
        # 1. stored computed cols, resolved
        stored_computed_cols = [c for c in view.cols if c.is_stored and c.is_computed]
        select_list = [c.value_expr.copy().resolve_computed_cols(unstored_only=False) for c in stored_computed_cols]

        # 2. embeddings for indices:
        # we include embeddings for indices by constructing computed cols; we need to do that for all indexed cols,
        # not just the stored ones
        indexed_cols = [c for c in view.cols if c.is_indexed]
        indexed_col_refs = [exprs.ColumnRef(c) for c in indexed_cols]
        from pixeltable.functions.image_embedding import openai_clip
        # explicitly resize images to the required size
        target_img_type = next(iter(openai_clip.md.signature.parameters.values()))
        select_list.extend([openai_clip(col_ref.resize(target_img_type.size)) for col_ref in indexed_col_refs])
        plan, select_list = cls.create_query_plan(
            view.base, select_list=select_list, where_clause=view.predicate, with_pk=True, ignore_errors=False,
            version=base_version)
        num_idx_cols = len(indexed_cols)
        db_col_info = [
            ColumnInfo(c, e.slot_idx)
            for c, e in zip(stored_computed_cols, select_list[:len(select_list) - num_idx_cols])
        ]
        idx_col_info = [ColumnInfo(c, e.slot_idx) for c, e in zip(indexed_cols, select_list[-num_idx_cols:])]
        return plan, db_col_info, idx_col_info, len(select_list)

    class AnalysisInfo:
        def __init__(self, tbl: catalog.Table):
            self.tbl = tbl
            # we need to make copies of the select list and group-by clause to avoid reusing past execution state
            self.select_list: List[exprs.Expr] = []
            self.group_by_clause: List[exprs.Expr] = []
            self.order_by_clause: List[Tuple[exprs.Expr, bool]] = []  # List[(expr, asc)]

            # all exprs that need to be evaluated
            self.all_exprs: List[exprs.Expr] = []
            # exprs that can be expressed via SQL and are retrieved directly from the store
            self.sql_exprs = exprs.UniqueExprList()
            # Where clause of the Select stmt of the SQL scan
            self.sql_where_clause: Optional[sql.sql.expression.ClauseElement] = None
            # filter predicate applied to output rows of the SQL scan
            self.filter: Optional[exprs.Predicate] = None
            self.similarity_clause: Optional[exprs.ImageSimilarityPredicate] = None
            self.agg_fn_calls: List[exprs.FunctionCall] = []
            self.agg_order_by: List[exprs.Expr] = []

    @classmethod
    def _is_agg_fn_call(cls, e: exprs.Expr) -> bool:
        return isinstance(e, exprs.FunctionCall) and e.is_agg_fn_call

    @classmethod
    def _determine_agg_status(cls, e: exprs.Expr, grouping_expr_idxs: Set[int]) -> Tuple[bool, bool]:
        """Determine whether expr is the input to or output of an aggregate function.
        Returns:
            (<is output>, <is input>)
        """
        if e.slot_idx in grouping_expr_idxs:
            return True, True
        elif cls._is_agg_fn_call(e):
            for c in e.components:
                _, is_input = cls._determine_agg_status(c, grouping_expr_idxs)
                if not is_input:
                    raise exc.Error(f'Invalid nested aggregates: {e}')
            return True, False
        elif isinstance(e, exprs.Literal):
            return True, True
        elif isinstance(e, exprs.ColumnRef):
            # we already know that this isn't a grouping expr
            return False, True
        else:
            # an expression such as <grouping expr 1> + <grouping expr 2> can both be the output and input of agg
            assert len(e.components) > 0
            component_is_output, component_is_input = zip(
                *[cls._determine_agg_status(c, grouping_expr_idxs) for c in e.components])
            is_output = component_is_output.count(True) == len(e.components)
            is_input = component_is_input.count(True) == len(e.components)
            if not is_output and not is_input:
                raise exc.Error(f'Invalid expression, mixes aggregate with non-aggregate: {e}')
            return is_output, is_input

    @classmethod
    def _analyze_query(
            cls, tbl: catalog.TableVersion, select_list: List[exprs.Expr],
            where_clause: Optional[exprs.Predicate] = None, group_by_clause: List[exprs.Expr] = [],
            order_by_clause: List[Tuple[exprs.Expr, bool]] = []
    ) -> AnalysisInfo:
        """Performs semantic analysis of query and returns AnalysisInfo"""
        info = cls.AnalysisInfo(tbl)
        # create copies to avoid reusing past execution state and remove references to unstored computed cols
        info.select_list = [e.copy().resolve_computed_cols(unstored_only=True) for e in select_list]
        if where_clause is not None:
            where_clause = where_clause.copy().resolve_computed_cols(unstored_only=True)
        info.group_by_clause = [e.copy().resolve_computed_cols(unstored_only=True) for e in group_by_clause]
        info.order_by_clause = [(e.copy().resolve_computed_cols(unstored_only=True), asc) for e, asc in order_by_clause]

        if where_clause is not None:
            info.sql_where_clause, info.filter = where_clause.extract_sql_predicate()
            if info.filter is not None:
                similarity_clauses, info.filter = info.filter.split_conjuncts(
                    lambda e: isinstance(e, exprs.ImageSimilarityPredicate))
                if len(similarity_clauses) > 1:
                    raise exc.Error(f'More than one nearest() not supported')
                if len(similarity_clauses) == 1:
                    if len(info.order_by_clause) > 0:
                        raise exc.Error((
                            f'nearest() returns results in order of proximity and cannot be used in conjunction with '
                            f'order_by()'))
                    info.similarity_clause = similarity_clauses[0]
                    img_col = info.similarity_clause.img_col_ref.col
                    if not img_col.is_indexed:
                        raise exc.Error(f'nearest() not available for unindexed column {img_col.name}')


        info.all_exprs = info.select_list.copy()
        info.all_exprs.extend(info.group_by_clause)
        info.all_exprs.extend([e for e, _ in info.order_by_clause])
        if info.filter is not None:
            info.all_exprs.append(info.filter)
        info.sql_exprs.extend(
            exprs.Expr.list_subexprs(info.all_exprs, filter=lambda e: e.sql_expr() is not None, traverse_matches=False))
        # we don't want to materialize literals via SQL, so we remove them here
        info.sql_exprs = exprs.UniqueExprList([e for e in info.sql_exprs if not isinstance(e, exprs.Literal)])
        return info

    @classmethod
    def _analyze_agg(cls, evaluator: exprs.Evaluator, info: AnalysisInfo) -> None:
        """Check semantic correctness of aggregation and fill in agg-specific fields of AnalysisInfo"""
        info.agg_fn_calls = [
            e for e in evaluator.unique_exprs if isinstance(e, exprs.FunctionCall) and e.is_agg_fn_call
        ]
        if len(info.agg_fn_calls) == 0:
            # nothing to do
            return

        # check that select list only contains aggregate output
        grouping_expr_idxs = [e.slot_idx for e in info.group_by_clause]
        is_agg_output = [cls._determine_agg_status(e, grouping_expr_idxs)[0] for e in info.select_list]
        if is_agg_output.count(False) > 0:
            raise exc.Error(
                f'Invalid non-aggregate expression in aggregate query: {info.select_list[is_agg_output.index(False)]}')

        # check that filter doesn't contain aggregates
        if info.filter is not None:
            agg_fn_calls = [e for e in info.filter.subexprs(filter=lambda e: cls._is_agg_fn_call(e))]
            if len(agg_fn_calls) > 0:
                raise exc.Error(f'Filter cannot contain aggregate functions: {info.filter}')

        # check that grouping exprs don't contain aggregates and can be expressed as SQL (we perform sort-based
        # aggregation and rely on the SqlScanNode returning data in the correct order)
        for e in info.group_by_clause:
            if e.sql_expr() is None:
                raise exc.Error(f'Invalid grouping expression, needs to be expressible in SQL: {e}')
            if e.contains(filter=lambda e: cls._is_agg_fn_call(e)):
                raise exc.Error(f'Grouping expression contains aggregate function: {e}')

        # check that agg fn calls don't have contradicting ordering requirements
        order_by: List[exprs.Exprs] = []
        order_by_origin: Optional[exprs.Expr] = None  # the expr that determines the ordering
        for agg_fn_call in info.agg_fn_calls:
            fn_call_order_by = agg_fn_call.get_agg_order_by()
            if len(fn_call_order_by) == 0:
                continue
            if len(order_by) == 0:
                order_by = fn_call_order_by
                order_by_origin = agg_fn_call
            else:
                combined = cls._get_combined_ordering(
                    [(e, True) for e in order_by], [(e, True) for e in fn_call_order_by])
                if len(combined) == 0:
                    raise exc.Error((
                        f"Incompatible ordering requirements between expressions '{order_by_origin}' and "
                        f"'{agg_fn_call}':\n"
                        f"{exprs.Expr.print_list(order_by)} vs {exprs.Expr.print_list(fn_call_order_by)}"
                    ))
        info.agg_order_by = order_by

    @classmethod
    def _get_combined_ordering(
            cls, o1: List[Tuple[exprs.Expr, bool]], o2: List[Tuple[exprs.Expr, bool]]
    ) -> List[Tuple[exprs.Expr, bool]]:
        """Returns an ordering that's compatible with both o1 and o2, or an empty list if no such ordering exists"""
        result: List[Tuple[exprs.Expr, bool]] = []
        # determine combined ordering
        for (e1, asc1), (e2, asc2) in zip(o1, o2):
            if e1.slot_idx != e2.slot_idx:
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

    @classmethod
    def _determine_ordering(
        cls, tbl: catalog.TableVersion, evaluator: exprs.Evaluator, info: AnalysisInfo
    ) -> List[sql.sql.expression.ClauseElement]:
        """Returns the ORDER BY clause of the SqlScanNode"""
        order_by_items: List[Tuple[exprs.Expr, bool]] = []
        order_by_origin: Optional[exprs.Expr] = None  # the expr that determines the ordering
        if evaluator.unique_exprs.contains(exprs.FrameColumnRef):
            # we're materializing extracted frames and need to order by the frame src and idx cols;
            # make sure to get the exprs from the evaluator, which have slot_idx set
            frame_src_col_ref = evaluator.unique_exprs[exprs.ColumnRef(tbl.frame_src_col())]
            frame_idx_col_ref = evaluator.unique_exprs[exprs.ColumnRef(tbl.frame_idx_col())]
            order_by_items = [(frame_src_col_ref, None), (frame_idx_col_ref, True)]  # need frame idx to be ascending
            order_by_origin = [e for e in evaluator.unique_exprs if isinstance(e, exprs.FrameColumnRef)][0]

        # window functions require ordering by the group_by/order_by clauses
        window_fn_calls = [
            e for e in evaluator.unique_exprs if isinstance(e, exprs.FunctionCall) and e.is_window_fn_call
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
                    combined = cls._get_combined_ordering(order_by_items, other_order_by_clauses)
                    if len(combined) == 0:
                        raise exc.Error((
                            f"Incompatible ordering requirements between expressions '{order_by_origin}' and "
                            f"'{fn_call}':\n"
                            f"{exprs.Expr.print_list(order_by_items)} vs {exprs.Expr.print_list(other_order_by_clauses)}"
                        ))
                    order_by_items = combined

        if len(info.group_by_clause) > 0:
            agg_ordering = [(e, None) for e in info.group_by_clause] + [(e, True) for e in info.agg_order_by]
            if len(order_by_items) > 0:
                # check for compatibility
                combined = cls._get_combined_ordering(order_by_items, agg_ordering)
                if len(combined) == 0:
                    raise exc.Error((
                        f"Incompatible ordering requirements between expressions '{order_by_origin}' and "
                        f"grouping expressions:\n"
                        f"{exprs.Expr.print_list([e for e, _ in order_by_items])} vs "
                        f"{exprs.Expr.print_list([e for e, _ in agg_ordering])}"
                    ))
                order_by_items = combined
            else:
                order_by_items = agg_ordering

        if len(info.order_by_clause) > 0:
            if len(order_by_items) > 0:
                # check for compatibility
                combined = cls._get_combined_ordering(order_by_items, info.order_by_clause)
                if len(combined) == 0:
                    raise exc.Error((
                        f"Incompatible ordering requirements between expressions '{order_by_origin}' and "
                        f"order-by expressions:\n"
                        f"{exprs.Expr.print_list([e for e, _ in order_by_items])} vs "
                        f"{exprs.Expr.print_list([e for e, _ in info.order_by_clause])}"
                    ))
                order_by_items = combined
            else:
                order_by_items = info.order_by_clause

        # we do ascending ordering by default, if not specified otherwise
        order_by_clause = [e.sql_expr().desc() if asc == False else e.sql_expr() for e, asc in order_by_items]
        for i in range(len(order_by_items)):
            if order_by_clause[i] is None:
                raise exc.Error(f'order_by element cannot be expressed in SQL: {order_by_items[i]}')
        return order_by_clause

    @classmethod
    def _is_contained_in(cls, l1: List[exprs.Expr], l2: List[exprs.Expr]) -> bool:
        """Returns True if l1 is contained in l2"""
        s1, s2 = set([e.slot_idx for e in l1]), set([e.slot_idx for e in l2])
        return s1 <= s2

    @classmethod
    def _insert_prefetch_node(
            cls, tbl_id: UUID, output_exprs: List[exprs.Expr], evaluator: exprs.Evaluator, input: ExecNode) -> ExecNode:
        """Returns a CachePrefetchNode into the plan if needed, otherwise returns input"""
        eval_ctx = evaluator.get_eval_ctx(output_exprs)
        media_col_refs = [
            e for e in eval_ctx
            if isinstance(e, exprs.ColumnRef) and (e.col_type.is_image_type() or e.col_type.is_video_type())
        ]
        if len(media_col_refs) == 0:
            return input
        # we need to prefetch external files for media column types
        file_col_info = [ColumnInfo(e.col, e.slot_idx) for e in media_col_refs]
        prefetch_node = CachePrefetchNode(tbl_id, file_col_info, input)
        return prefetch_node

    @classmethod
    def create_query_plan(
            cls, tbl: catalog.TableVersion, select_list: List[exprs.Expr],
            where_clause: Optional[exprs.Predicate] = None, group_by_clause: List[exprs.Expr] = [],
            order_by_clause: List[Tuple[exprs.Expr, bool]] = [], limit: Optional[int] = None,
            with_pk: bool = False, ignore_errors: bool = False, version: Optional[int] = None
    ) -> Tuple[ExecNode, List[exprs.Expr]]:
        info = cls._analyze_query(
            tbl, select_list, where_clause=where_clause, group_by_clause=group_by_clause,
            order_by_clause=order_by_clause)
        evaluator = exprs.Evaluator(info.all_exprs, info.sql_exprs)
        cls._analyze_agg(evaluator, info)
        is_agg_query = len(info.group_by_clause) > 0 or len(info.agg_fn_calls) > 0
        ctx = ExecContext(evaluator)

        order_by_clause = cls._determine_ordering(tbl, evaluator, info)
        sql_limit = 0 if is_agg_query else limit  # if we're aggregating, the limit applies to the agg output
        plan = SqlScanNode(
            tbl, evaluator, info.sql_exprs, where_clause=info.sql_where_clause, filter=info.filter, limit=sql_limit,
            order_by_clause=order_by_clause, set_pk=True, similarity_clause=info.similarity_clause, version=version)
            #order_by_clause = order_by_clause, set_pk = with_pk, similarity_clause = info.similarity_clause, version = version)
        plan = cls._insert_prefetch_node(tbl.id, info.select_list, evaluator, plan)

        if len(info.group_by_clause) > 0 or len(info.agg_fn_calls) > 0:
            # we're doing aggregation; the input of the AggregateNode are the grouping exprs plus the
            # args of the agg fn calls
            agg_input = exprs.UniqueExprList(info.group_by_clause.copy())
            for fn_call in info.agg_fn_calls:
                agg_input.extend(fn_call.components)
            if not cls._is_contained_in(agg_input, info.sql_exprs):
                # we need an ExprEvalNode
                plan = ExprEvalNode(evaluator, agg_input, info.sql_exprs, ignore_errors=ignore_errors, input=plan)

            # batch size for aggregation input: this could be the entire table, so we need to divide it into
            # smaller batches; at the same time, we need to make the batches large enough to amortize the
            # function call overhead
            # TODO: increase this if we have NOS calls in order to reduce the cost of switching models, but take
            # into account the amount of memory needed for intermediate images
            ctx.batch_size = 16

            plan = AggregationNode(tbl, evaluator, info.group_by_clause, info.agg_fn_calls, agg_input, input=plan)
            agg_output = info.group_by_clause + info.agg_fn_calls
            if not cls._is_contained_in(info.select_list, agg_output):
                # we need an ExprEvalNode to evaluate the remaining output exprs
                plan = ExprEvalNode(evaluator, info.select_list, agg_output, ignore_errors=ignore_errors, input=plan)
        else:
            if not cls._is_contained_in(info.select_list, info.sql_exprs):
                # we need an ExprEvalNode to evaluate the remaining output exprs
                plan = ExprEvalNode(
                    evaluator, info.select_list, info.sql_exprs, ignore_errors=ignore_errors, input=plan)
            # we're returning everything to the user, so we might as well do it in a single batch
            ctx.batch_size = 0
        plan.set_ctx(ctx)

        return plan, info.select_list

    @classmethod
    def get_info(cls, tbl: catalog.TableVersion, where_clause: exprs.Predicate) -> AnalysisInfo:
        return cls._analyze_query(tbl, [], where_clause=where_clause)

    @classmethod
    def create_add_column_plan(
            cls, tbl: catalog.TableVersion, col: catalog.Column) -> Tuple[ExecNode, Optional[int], Optional[int]]:
        """Creates a plan for MutableTable.add_column()
        Returns:
            plan: the plan to execute
            ctx: the context to use for the plan
            value_expr slot idx for the plan output (for computed cols)
            embedding slot idx for the plan output (for indexed image cols)
        """
        select_list = []
        value_expr_pos: Optional[int] = None
        embedding_pos: Optional[int] = None
        if col.is_computed:
            select_list.append(col.value_expr)
            value_expr_pos = 0
        if col.is_indexed:
            # we also need to compute the embeddings for this col
            assert col.col_type.is_image_type()
            from pixeltable.functions.image_embedding import openai_clip
            select_list.append(openai_clip(col.value_expr))
            embedding_pos = len(select_list) - 1
        plan, select_list = cls.create_query_plan(tbl, select_list, with_pk=True, ignore_errors=True)
        assert len(select_list) <= 2
        plan.ctx.batch_size = 16
        plan.ctx.show_pbar = True

        # we want to flush images
        col_slot_idx = select_list[value_expr_pos].slot_idx if value_expr_pos is not None else None
        stored_col_info: List[ColumnInfo] = []
        if col.is_computed and col.is_stored and col.col_type.is_image_type():
            stored_col_info = [ColumnInfo(col, col_slot_idx)]
        plan.set_stored_img_cols(stored_col_info)
        return plan, \
            select_list[value_expr_pos].slot_idx if value_expr_pos is not None else None, \
            select_list[embedding_pos].slot_idx if embedding_pos is not None else None
