from __future__ import annotations

import logging
import sys
from typing import Any, AsyncIterator, Iterable, Optional, cast

from pixeltable import catalog, exceptions as excs, exprs

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class AggregationNode(ExecNode):
    """
    In-memory aggregation for UDAs.

    At the moment, this returns all results in a single DataRowBatch.
    """

    group_by: Optional[list[exprs.Expr]]
    input_exprs: list[exprs.Expr]
    agg_fn_eval_ctx: exprs.RowBuilder.EvalCtx
    agg_fn_calls: list[exprs.FunctionCall]
    output_batch: DataRowBatch
    limit: Optional[int]

    def __init__(
        self,
        tbl: catalog.TableVersionHandle,
        row_builder: exprs.RowBuilder,
        group_by: Optional[list[exprs.Expr]],
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
        # create output_batch here, rather than in __iter__(), so we don't need to remember tbl and row_builder
        self.output_batch = DataRowBatch(tbl, row_builder, 0)
        self.limit = None

    def set_limit(self, limit: int) -> None:
        # we can't propagate the limit to our input
        self.limit = limit

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

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        prev_row: Optional[exprs.DataRow] = None
        current_group: Optional[list[Any]] = None  # the values of the group-by exprs
        num_input_rows = 0
        num_output_rows = 0
        async for row_batch in self.input:
            num_input_rows += len(row_batch)
            for row in row_batch:
                group = [row[e.slot_idx] for e in self.group_by] if self.group_by is not None else None

                if current_group is None:
                    current_group = group
                    self._reset_agg_state(0)

                if group != current_group:
                    # we're entering a new group, emit a row for the previous one
                    self.row_builder.eval(prev_row, self.agg_fn_eval_ctx, profile=self.ctx.profile)
                    self.output_batch.add_row(prev_row)
                    num_output_rows += 1
                    if self.limit is not None and num_output_rows == self.limit:
                        yield self.output_batch
                        return
                    current_group = group
                    self._reset_agg_state(0)
                self._update_agg_state(row, 0)
                prev_row = row

        if prev_row is not None:
            # emit the last group
            self.row_builder.eval(prev_row, self.agg_fn_eval_ctx, profile=self.ctx.profile)
            self.output_batch.add_row(prev_row)

        self.output_batch.flush_imgs(None, self.stored_img_cols, self.flushed_img_slots)
        _logger.debug(f'AggregateNode: consumed {num_input_rows} rows, returning {len(self.output_batch.rows)} rows')
        yield self.output_batch
