from __future__ import annotations

import contextlib
import sys
from typing import Any, AsyncGenerator, Sequence, cast

from pixeltable import exceptions as excs, exprs, func

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode


# TODO move to a util module?
def _hashable(val: Any) -> Any:
    """A hashable value that is equal for equal JSON or scalar vals, for use in a dict key."""
    if isinstance(val, dict):
        return frozenset((k, _hashable(v)) for k, v in val.items())
    if isinstance(val, list):
        return tuple(_hashable(v) for v in val)
    return val


class WindowPartitionNode(ExecNode):
    """
    Evaluates window function calls without order_by: each row gets the aggregate over its entire partition.

    Rows are grouped by their partition key in a dict. We can't rely on the input to be already sorted by group_by
    exprs because that would clash with sample(), head(), tail(), or order_by(), and SQL can't sort by group_by exprs
    that have no SQL translation.

    Accumulates all rows in memory. That doesn't add memory pressure because non-aggregate queries already return their
    entire result in a single batch.
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

    @classmethod
    def validate_group_by(cls, fn_call: exprs.FunctionCall) -> None:
        for e in fn_call.group_by:
            # the group_by values are used as dict keys, and arrays and images aren't hashable
            if e.col_type.is_array_type() or e.col_type.is_image_type():
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Window functions without order_by cannot have an array or image group_by: {e!s}',
                )

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

    def _agg_value(self, fn_call: exprs.FunctionCall, aggregator: func.Aggregator, row_num: int) -> Any:
        try:
            return aggregator.value()
        except Exception as exc:
            _, _, exc_tb = sys.exc_info()
            raise excs.ExprEvalError(
                fn_call, f'value() function of the aggregate {fn_call!s}', exc, exc_tb, [], row_num
            ) from exc

    def _partition_key(self, fn_call: exprs.FunctionCall, row: exprs.DataRow) -> Any:
        return tuple(_hashable(row[e.slot_idx]) for e in fn_call.group_by)

    def _eval_fn_call(
        self, fn_call: exprs.FunctionCall, rows: list[exprs.DataRow], output_rows: list[exprs.DataRow]
    ) -> None:
        # partition key -> (input position of the partition's first row, aggregator over the partition's rows)
        partitions: dict[Any, tuple[int, func.Aggregator]] = {}
        for row_num, row in enumerate(rows):
            key = self._partition_key(fn_call, row)
            if key not in partitions:
                self._reset_agg(fn_call, row_num)
                partitions[key] = (row_num, fn_call.aggregator)
            fn_call.aggregator = partitions[key][1]
            self._update_agg(fn_call, row, row_num)
        vals = {key: self._agg_value(fn_call, agg, row_num) for key, (row_num, agg) in partitions.items()}
        for row in output_rows:
            row[fn_call.slot_idx] = exprs.defensive_copy(vals[self._partition_key(fn_call, row)])

    async def __aiter__(self) -> AsyncGenerator[DataRowBatch, None]:
        limit = self._resolve_positive_int(self.limit, 'limit') if self.limit is not None else None
        offset = self._resolve_positive_int(self.offset, 'offset') if self.offset is not None else 0
        if limit == 0:
            return

        rows: list[exprs.DataRow] = []
        async with contextlib.aclosing(aiter(self.input)) as input_iter:
            async for batch in input_iter:
                rows.extend(batch)

        # every row contributes to its partitions, but only the rows within limit and offset need the values
        output_rows = rows[offset:] if limit is None else rows[offset : offset + limit]
        if len(output_rows) == 0:
            return
        for fn_call in self.fn_calls:
            self._eval_fn_call(fn_call, rows, output_rows)
        yield DataRowBatch(self.row_builder, output_rows)
