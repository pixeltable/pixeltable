from __future__ import annotations

import logging
import sys
from typing import List, Optional, Any

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')

class AggregationNode(ExecNode):
    def __init__(
            self, tbl: catalog.TableVersion, row_builder: exprs.RowBuilder, group_by: List[exprs.Expr],
            agg_fn_calls: List[exprs.FunctionCall], input_exprs: List[exprs.Expr], input: ExecNode
    ):
        super().__init__(row_builder, group_by + agg_fn_calls, input_exprs, input)
        self.input = input
        self.group_by = group_by
        self.input_exprs = input_exprs
        self.agg_fn_calls = agg_fn_calls
        self.agg_fn_eval_ctx = row_builder.create_eval_ctx(agg_fn_calls, exclude=input_exprs)
        self.output_batch = DataRowBatch(tbl, row_builder, 0)

    def _reset_agg_state(self, row_num: int) -> None:
        for fn_call in self.agg_fn_calls:
            try:
                fn_call.reset_agg()
            except Exception as e:
                _, _, exc_tb = sys.exc_info()
                expr_msg = f'init() function of the aggregate {fn_call}'
                raise excs.ExprEvalError(fn_call, expr_msg, e, exc_tb, [], row_num)

    def _update_agg_state(self, row: exprs.DataRow, row_num: int) -> None:
        for fn_call in self.agg_fn_calls:
            try:
                fn_call.update(row)
            except Exception as e:
                _, _, exc_tb = sys.exc_info()
                expr_msg = f'update() function of the aggregate {fn_call}'
                input_vals = [row[d.slot_idx] for d in fn_call.dependencies()]
                raise excs.ExprEvalError(fn_call, expr_msg, e, exc_tb, input_vals, row_num)

    def __next__(self) -> DataRowBatch:
        if self.output_batch is None:
            raise StopIteration

        prev_row: Optional[exprs.DataRow] = None
        current_group: Optional[List[Any]] = None  # the values of the group-by exprs
        num_input_rows = 0
        for row_batch in self.input:
            num_input_rows += len(row_batch)
            for row in row_batch:
                group = [row[e.slot_idx] for e in self.group_by]
                if current_group is None:
                    current_group = group
                    self._reset_agg_state(0)
                if group != current_group:
                    # we're entering a new group, emit a row for the previous one
                    self.row_builder.eval(prev_row, self.agg_fn_eval_ctx, profile=self.ctx.profile)
                    self.output_batch.add_row(prev_row)
                    current_group = group
                    self._reset_agg_state(0)
                self._update_agg_state(row, 0)
                prev_row = row
        # emit the last group
        self.row_builder.eval(prev_row, self.agg_fn_eval_ctx, profile=self.ctx.profile)
        self.output_batch.add_row(prev_row)

        result = self.output_batch
        result.flush_imgs(None, self.stored_img_cols, self.flushed_img_slots)
        self.output_batch = None
        _logger.debug(f'AggregateNode: consumed {num_input_rows} rows, returning {len(result.rows)} rows')
        return result

