from __future__ import annotations

import logging
import sys
from typing import Any, AsyncIterator, Iterable, cast

from pixeltable import catalog, exceptions as excs, exprs, hooks

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger(__name__)


class AggregationNode(ExecNode):
    """
    In-memory aggregation for UDAs.

    At the moment, this returns all results in a single DataRowBatch.
    """

    group_by: list[exprs.Expr] | None
    input_exprs: list[exprs.Expr]
    agg_fn_eval_ctx: exprs.RowBuilder.EvalCtx
    agg_fn_calls: list[exprs.FunctionCall]
    output_batch: DataRowBatch
    limit: exprs.Expr | None

    def __init__(
        self,
        tbl: catalog.TableVersionHandle,
        row_builder: exprs.RowBuilder,
        group_by: list[exprs.Expr] | None,
        agg_fn_calls: list[exprs.FunctionCall],
        input_exprs: Iterable[exprs.Expr],
        input: ExecNode,
    ):
        output_exprs: list[exprs.Expr] = [] if group_by is None else list(group_by)
        output_exprs.extend(agg_fn_calls)
        super().__init__(row_builder, output_exprs, input_exprs, input)
        self.input = input
        self.group_by = group_by
        self.input_exprs = list(input_exprs)
        self.agg_fn_eval_ctx = row_builder.create_eval_ctx(agg_fn_calls, exclude=self.input_exprs)
        # we need to make sure to refer to the same exprs that RowBuilder.eval() will use
        self.agg_fn_calls = [cast(exprs.FunctionCall, e) for e in self.agg_fn_eval_ctx.target_exprs]
        self.limit = None
        self._init_exec_state()

    def _init_exec_state(self) -> None:
        self.output_batch = DataRowBatch(self.row_builder)

    def _open(self) -> None:
        self._init_exec_state()

    def set_limit(self, limit: exprs.Expr) -> None:
        # we can't propagate the limit to our input
        self.limit = limit

    def init_bindings(self) -> None:
        self.bind_sources = (self.group_by or []) + list(self.agg_fn_calls)
        if self.limit is not None:
            self.bind_sources.append(self.limit)
        super().init_bindings()

    def _reset_agg_state(self, row_num: int) -> None:
        for fn_call in self.agg_fn_calls:
            try:
                fn_call.reset_agg()
            except Exception as exc:
                _, _, exc_tb = sys.exc_info()
                expr_msg = f'init() function of the aggregate {fn_call}'
                raise excs.ExprEvalError(fn_call, expr_msg, exc, exc_tb, [], row_num) from exc

    def _update_agg_state(self, row: exprs.DataRow, row_num: int) -> None:
        for fn_call in self.agg_fn_calls:
            try:
                fn_call.update(row)
            except Exception as exc:
                _, _, exc_tb = sys.exc_info()
                expr_msg = f'update() function of the aggregate {fn_call}'
                input_vals = [row[d.slot_idx] for d in fn_call.dependencies()]
                raise excs.ExprEvalError(fn_call, expr_msg, exc, exc_tb, input_vals, row_num) from exc

    @property
    def _agg_name(self) -> str:
        return self.agg_fn_calls[0].fn.display_name if len(self.agg_fn_calls) > 0 else 'agg'

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        limit = self._resolve_positive_int(self.limit, 'limit') if self.limit is not None else None
        if limit == 0:
            return
        prev_row: exprs.DataRow | None = None
        current_group: list[Any] | None = None  # the values of the group-by exprs
        num_input_rows = 0
        num_output_rows = 0
        async for row_batch in self.input:
            self.set_var_slots(row_batch)
            num_input_rows += len(row_batch)
            for row in row_batch:
                group = [row[e.slot_idx] for e in self.group_by] if self.group_by is not None else None

                if current_group is None:
                    current_group = group
                    self._reset_agg_state(0)

                if group != current_group:
                    # we're entering a new group, emit a row for the previous one
                    with hooks.span(f'pixeltable.agg.{self._agg_name}', level=hooks.DEBUG, parent=self._span):
                        self.row_builder.eval(prev_row, self.agg_fn_eval_ctx, profile=self.ctx.profile)
                    self.output_batch.add_row(prev_row)
                    num_output_rows += 1
                    if limit is not None and num_output_rows == limit:
                        yield self.output_batch
                        return
                    current_group = group
                    self._reset_agg_state(0)
                self._update_agg_state(row, 0)
                prev_row = row

        if prev_row is not None:
            # emit the last group
            with hooks.span(f'pixeltable.agg.{self._agg_name}', level=hooks.DEBUG, parent=self._span):
                self.row_builder.eval(prev_row, self.agg_fn_eval_ctx, profile=self.ctx.profile)
            self.output_batch.add_row(prev_row)

        _logger.debug(f'AggregateNode: consumed {num_input_rows} rows, returning {len(self.output_batch.rows)} rows')
        yield self.output_batch
