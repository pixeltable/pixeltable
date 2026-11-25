from __future__ import annotations

import contextlib
import sys
from typing import Any, AsyncGenerator, Sequence, cast

from pixeltable import exceptions as excs, exprs

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode


class WindowPartitionNode(ExecNode):
    """
    Evaluates window function calls without order_by: each row gets the aggregate over its entire partition.

    The input needs to be ordered by the group_by exprs of every call. A row is held back until the partitions of all
    calls that contain it are complete.
    """

    fn_calls: list[exprs.FunctionCall]
    limit: exprs.Expr | None
    offset: exprs.Expr | None

    def __init__(self, row_builder: exprs.RowBuilder, fn_calls: Sequence[exprs.FunctionCall], input: ExecNode):
        super().__init__(row_builder, [], [], input)
        assert all(fn_call.is_partition_window_fn_call for fn_call in fn_calls)
        self.fn_calls = [cast(exprs.FunctionCall, row_builder.unique_exprs[fn_call]) for fn_call in fn_calls]
        self.limit = None
        self.offset = None

    def set_limit(self, limit: exprs.Expr) -> None:
        self.limit = limit

    def set_offset(self, offset: exprs.Expr) -> None:
        self.offset = offset

    def init_bindings(self) -> None:
        self.bind_sources = [e for e in (self.limit, self.offset) if e is not None]
        super().init_bindings()

    def _reset_agg(self, fn_call: exprs.FunctionCall, row_num: int) -> None:
        try:
            fn_call.reset_agg()
        except Exception as exc:
            _, _, exc_tb = sys.exc_info()
            raise excs.ExprEvalError(
                fn_call, f'init() function of the aggregate {fn_call!s}', exc, exc_tb, [], row_num
            ) from exc

    def _update_agg(self, fn_call: exprs.FunctionCall, row: exprs.DataRow, row_num: int) -> None:
        try:
            fn_call.update(row)
        except Exception as exc:
            _, _, exc_tb = sys.exc_info()
            input_vals = [row[d.slot_idx] for d in fn_call.dependencies()]
            raise excs.ExprEvalError(
                fn_call, f'update() function of the aggregate {fn_call!s}', exc, exc_tb, input_vals, row_num
            ) from exc

    def _set_partition_value(self, fn_call: exprs.FunctionCall, rows: list[exprs.DataRow], row_num: int) -> None:
        try:
            val = fn_call.aggregator.value()
        except Exception as exc:
            _, _, exc_tb = sys.exc_info()
            raise excs.ExprEvalError(
                fn_call, f'value() function of the aggregate {fn_call!s}', exc, exc_tb, [], row_num
            ) from exc
        for i, row in enumerate(rows):
            row[fn_call.slot_idx] = val if i == 0 else exprs.defensive_copy(val)

    async def __aiter__(self) -> AsyncGenerator[DataRowBatch, None]:
        limit = self._resolve_positive_int(self.limit, 'limit') if self.limit is not None else None
        offset = self._resolve_positive_int(self.offset, 'offset') if self.offset is not None else 0
        if limit == 0:
            return

        pending: list[exprs.DataRow] = []
        pending_start = 0
        partition_starts = [0] * len(self.fn_calls)
        partition_vals: list[list[Any]] = [[] for _ in self.fn_calls]
        num_input_rows = 0
        num_completed = 0
        num_returned = 0

        def take_completed(end: int) -> DataRowBatch:
            nonlocal pending_start, num_completed, num_returned
            output_batch = DataRowBatch(self.row_builder)
            for row in pending[: end - pending_start]:
                num_completed += 1
                if num_completed <= offset:
                    continue
                if limit is not None and num_returned == limit:
                    break
                output_batch.add_row(row)
                num_returned += 1
            del pending[: end - pending_start]
            pending_start = end
            return output_batch

        async with contextlib.aclosing(aiter(self.input)) as input_iter:
            async for batch in input_iter:
                for row in batch:
                    for i, fn_call in enumerate(self.fn_calls):
                        vals = [row[e.slot_idx] for e in fn_call.group_by]
                        if num_input_rows == 0:
                            self._reset_agg(fn_call, num_input_rows)
                        elif vals != partition_vals[i]:
                            rows = pending[partition_starts[i] - pending_start :]
                            self._set_partition_value(fn_call, rows, num_input_rows)
                            self._reset_agg(fn_call, num_input_rows)
                            partition_starts[i] = num_input_rows
                        partition_vals[i] = vals
                        self._update_agg(fn_call, row, num_input_rows)
                    pending.append(row)
                    num_input_rows += 1

                output_batch = take_completed(min(partition_starts))
                if len(output_batch) > 0:
                    yield output_batch
                if limit is not None and num_returned == limit:
                    return

        if num_input_rows == 0:
            return
        for i, fn_call in enumerate(self.fn_calls):
            self._set_partition_value(fn_call, pending[partition_starts[i] - pending_start :], num_input_rows)
        output_batch = take_completed(num_input_rows)
        if len(output_batch) > 0:
            yield output_batch
