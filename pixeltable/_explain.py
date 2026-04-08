"""Query execution plan visualization for Pixeltable.

Provides human-readable representations of ExecNode trees, similar to SQL EXPLAIN.
"""

from __future__ import annotations

from collections.abc import Callable

from pixeltable import exec, exprs


def _collect_nodes(node: exec.ExecNode) -> list[exec.ExecNode]:
    """Walk the ExecNode input chain and return nodes in data-flow order (source first).

    Note: SqlJoinNode, SqlAggregationNode, and SqlSampleNode consume their input SqlNodes
    via CTEs (Common Table Expressions) rather than the ExecNode.input chain. Those upstream
    scan nodes are not traversable here; the CTE-consuming node appears as the chain's leaf
    and its formatter shows the relevant join/aggregation/sample metadata.
    """
    nodes: list[exec.ExecNode] = []
    current = node
    while current is not None:
        nodes.append(current)
        current = current.input
    nodes.reverse()
    return nodes


def _expr_display(e: exprs.Expr) -> str:
    """Return a concise display string for an expression."""
    return e.display_str(inline=False)


def _format_sql_scan_node(node: exec.SqlScanNode) -> list[str]:
    lines: list[str] = []
    tbl_name = node.tbl.tbl_name() if node.tbl is not None else '<unknown>'
    lines.append(f'SqlScanNode [table: {tbl_name!r}]')
    lines.append(f'  output: {len(node.select_list)} expression(s)')
    if node.where_clause is not None:
        lines.append(f'  where: {_expr_display(node.where_clause)} [SQL]')
    if node.py_filter is not None:
        lines.append(f'  filter: {_expr_display(node.py_filter)} [Python]')
    if len(node.order_by_clause) > 0:
        ob_parts = []
        for e, asc in node.order_by_clause:
            direction = 'asc' if asc is not False else 'desc'
            ob_parts.append(f'{_expr_display(e)} {direction}')
        lines.append(f'  order_by: {", ".join(ob_parts)}')
    if node.limit is not None:
        limit_str = str(node.limit)
        if node.offset is not None:
            limit_str += f' offset {node.offset}'
        lines.append(f'  limit: {limit_str}')
    return lines


def _format_sql_join_node(node: exec.SqlJoinNode) -> list[str]:
    from pixeltable.plan import JoinType

    lines: list[str] = []
    join_types = []
    for jc in node.join_clauses:
        jt = jc.join_type
        if jt == JoinType.INNER:
            join_types.append('inner')
        elif jt == JoinType.LEFT:
            join_types.append('left')
        elif jt == JoinType.FULL_OUTER:
            join_types.append('full_outer')
        elif jt == JoinType.CROSS:
            join_types.append('cross')
        else:
            join_types.append(str(jt))
    lines.append(f'SqlJoinNode [{", ".join(join_types)} join, {len(node.input_ctes)} tables]')
    lines.append(f'  output: {len(node.select_list)} expression(s)')
    # Show join predicates
    for jc in node.join_clauses:
        if jc.join_predicate is not None:
            lines.append(f'  on: {_expr_display(jc.join_predicate)}')
    if node.where_clause is not None:
        lines.append(f'  where: {_expr_display(node.where_clause)} [SQL]')
    return lines


def _format_sql_aggregation_node(node: exec.SqlAggregationNode) -> list[str]:
    lines: list[str] = []
    lines.append('SqlAggregationNode [SQL]')
    lines.append(f'  output: {len(node.select_list)} expression(s)')
    if node.group_by_items is not None:
        gb_parts = [_expr_display(e) for e in node.group_by_items]
        lines.append(f'  group_by: {", ".join(gb_parts)}')
    return lines


def _format_sql_sample_node(node: exec.SqlSampleNode) -> list[str]:
    lines: list[str] = []
    sample = node.sample_clause
    if sample.n is not None:
        lines.append(f'SqlSampleNode [n={sample.n}]')
    elif sample.fraction is not None:
        lines.append(f'SqlSampleNode [fraction={sample.fraction}]')
    elif sample.n_per_stratum is not None:
        lines.append(f'SqlSampleNode [n_per_stratum={sample.n_per_stratum}]')
    else:
        lines.append('SqlSampleNode')
    if node.stratify_exprs:
        parts = [_expr_display(e) for e in node.stratify_exprs]
        lines.append(f'  stratify_by: {", ".join(parts)}')
    return lines


def _format_expr_eval_node(node: exec.ExprEvalNode) -> list[str]:
    lines: list[str] = []
    # Count the expressions being evaluated
    output_count = int(node.outputs.sum()) if hasattr(node, 'outputs') and node.outputs is not None else 0
    lines.append(f'ExprEvalNode [{output_count} expression(s), Python]')

    # Show function calls, especially API-backed ones
    for e in node.output_exprs:
        if isinstance(e, exprs.FunctionCall) and e.fn is not None:
            fn_name = e.fn.display_name if hasattr(e.fn, 'display_name') else str(e.fn)
            lines.append(f'  eval: {_expr_display(e)} [{fn_name}]')
    return lines


def _format_aggregation_node(node: exec.AggregationNode) -> list[str]:
    lines: list[str] = []
    num_fns = len(node.agg_fn_calls)
    lines.append(f'AggregationNode [{num_fns} aggregate function(s), Python]')
    if node.group_by is not None:
        gb_parts = [_expr_display(e) for e in node.group_by]
        lines.append(f'  group_by: {", ".join(gb_parts)}')
    for fn_call in node.agg_fn_calls:
        lines.append(f'  agg: {_expr_display(fn_call)}')
    return lines


def _format_cache_prefetch_node(node: exec.CachePrefetchNode) -> list[str]:
    lines: list[str] = []
    num_cols = len(node.file_col_info)
    lines.append(f'CachePrefetchNode [{num_cols} media column(s)]')
    lines.append(f'  max_workers: {node.MAX_WORKERS}')
    return lines


def _format_cell_materialization_node(node: exec.CellMaterializationNode) -> list[str]:
    lines: list[str] = []
    lines.append('CellMaterializationNode')
    return lines


def _format_cell_reconstruction_node(node: exec.CellReconstructionNode) -> list[str]:
    lines: list[str] = []
    lines.append('CellReconstructionNode')
    return lines


def _format_object_store_save_node(node: exec.ObjectStoreSaveNode) -> list[str]:
    lines: list[str] = []
    lines.append('ObjectStoreSaveNode')
    lines.append(f'  max_workers: {node.MAX_WORKERS}')
    return lines


def _format_component_iteration_node(node: exec.ComponentIterationNode) -> list[str]:
    lines: list[str] = []
    iterator_name = str(node.iterator_call)
    lines.append(f'ComponentIterationNode [iterator: {iterator_name}]')
    return lines


def _format_generic_node(node: exec.ExecNode) -> list[str]:
    return [type(node).__name__]


def _format_sql_lookup_node(node: exec.SqlLookupNode) -> list[str]:
    lines: list[str] = []
    tbl_name = node.tbl.tbl_name() if node.tbl is not None else '<unknown>'
    lines.append(f'SqlLookupNode [table: {tbl_name!r}]')
    lines.append(f'  output: {len(node.select_list)} expression(s)')
    return lines


_NODE_FORMATTERS = {
    exec.SqlScanNode: _format_sql_scan_node,
    exec.SqlLookupNode: _format_sql_lookup_node,
    exec.SqlJoinNode: _format_sql_join_node,
    exec.SqlAggregationNode: _format_sql_aggregation_node,
    exec.SqlSampleNode: _format_sql_sample_node,
    exec.ExprEvalNode: _format_expr_eval_node,
    exec.AggregationNode: _format_aggregation_node,
    exec.CachePrefetchNode: _format_cache_prefetch_node,
    exec.CellMaterializationNode: _format_cell_materialization_node,
    exec.CellReconstructionNode: _format_cell_reconstruction_node,
    exec.ObjectStoreSaveNode: _format_object_store_save_node,
    exec.ComponentIterationNode: _format_component_iteration_node,
}


def format_exec_plan(root_node: exec.ExecNode) -> str:
    """Walk the ExecNode tree and produce a human-readable execution plan.

    The output shows the execution pipeline in data-flow order (source at top, output at bottom),
    with each node's key properties displayed.

    Args:
        root_node: The root (output) node of the execution plan.

    Returns:
        A formatted multi-line string describing the execution plan.
    """
    nodes = _collect_nodes(root_node)
    all_lines: list[str] = []

    for i, node in enumerate(nodes):
        formatter: Callable[..., list[str]] = _NODE_FORMATTERS.get(type(node), _format_generic_node)
        node_lines = formatter(node)

        if i > 0:
            # Add a connector arrow before non-first nodes
            all_lines.append(f'-> {node_lines[0]}')
            for line in node_lines[1:]:
                all_lines.append(f'   {line}')
        else:
            all_lines.extend(node_lines)

    return '\n'.join(all_lines)
