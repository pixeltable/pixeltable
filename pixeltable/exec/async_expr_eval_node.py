import abc
import asyncio
import logging
import sys
from typing import Iterable, Iterator, Optional, Any, AsyncIterator

import numpy as np

from pixeltable import exprs
from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class Scheduler:
    def __init__(self):
        pass

    @abc.abstractmethod
    async def schedule(self, rows: list[exprs.DataRow], slot_idx: int, dispatcher: Any) -> None:
        pass


class DefaultExprScheduler(Scheduler):
    e: exprs.Expr

    def __init__(self, e: exprs.Expr):
        self.e = e

    async def schedule(self, rows: list[exprs.DataRow], slot_idx: int, dispatch: Any) -> None:
        assert self.e.slot_idx >= 0
        for row in rows:
            assert not row.has_val[self.e.slot_idx] and not row.has_exc(self.e.slot_idx)
            try:
                self.e.eval(row, None)
            except Exception as exc:
                # TODO: propagate exceptions
                _, _, exc_tb = sys.exc_info()
        await dispatch(rows)


class AsyncExprEvalNode(ExecNode):
    num_dependencies: np.ndarray  # number of dependencies for our output slots; indexed by slot idx
    output_mask: np.ndarray
    output_rows: dict[int, Optional[exprs.DataRow]]  # key: id(row)
    slot_schedulers: dict[int, Scheduler]
    tasks: set[asyncio.Task]

    BATCH_SIZE = 12
    MAX_IN_FLIGHT = 50

    def __init__(
        self, row_builder: exprs.RowBuilder, output_exprs: Iterable[exprs.Expr], input_exprs: Iterable[exprs.Expr],
        input: ExecNode
    ):
        super().__init__(row_builder, output_exprs, input_exprs, input)
        self.num_dependencies = np.sum(row_builder.dependencies, axis=1)
        self.output_mask = np.zeros(row_builder.num_materialized, dtype=bool)
        output_slot_idxs = [e.slot_idx for e in output_exprs]
        self.output_mask[output_slot_idxs] = True
        # self.missing_dependencies = self.missing_dependencies * output_mask
        self.output_rows = {}
        self.slot_schedulers = {
            slot_idx: DefaultExprScheduler(row_builder.unique_exprs[slot_idx])
            for slot_idx in range(row_builder.num_materialized)
        }
        self.tasks = set()

    def __iter__(self) -> Iterator[DataRowBatch]:
        for batch in self.input:
            # for row in batch:
            #     np.copyto(row.missing_dependencies, self.missing_dependencies)
            # yield self.eval_batch(batch)

            # record rows in their original order
            self.output_rows = {id(row): None for row in batch}
            asyncio.run(self.dispatch(batch.rows))
            yield batch

    def __iter__(self) -> Iterator[DataRowBatch]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return iter(loop.run_until_complete(self._main_loop()))
        finally:
            loop.close()

    async def _main_loop(self) -> AsyncIterator[DataRowBatch]:
        pass

    async def dispatch(self, rows: list[exprs.DataRow]) -> None:
        """Dispatches row slots to the appropriate schedulers and returns immediately"""
        ready_slots = np.zeros((len(rows), self.row_builder.num_materialized), dtype=bool)
        for i, row in enumerate(rows):
            if np.array_equal(row.has_val * self.output_mask, self.output_mask):
                # all output slots have been materialized
                self.output_rows[id(row)] = row
            else:
                # determine ready slots that are not yet materialized
                num_mat_dependencies = np.sum(self.row_builder.dependencies * row.has_val, axis=1)
                num_missing = self.num_dependencies - num_mat_dependencies
                ready_slots[i] = (num_missing == 0) * (row.has_val == False) * self.output_mask

        # schedule all ready slots
        for slot_idx in np.sum(ready_slots, axis=0).nonzero()[0]:
            ready_rows_v = ready_slots[:, slot_idx].flatten()
            _ = ready_rows_v.nonzero()
            ready_rows = [rows[i] for i in ready_rows_v.nonzero()[0]]
            task = asyncio.create_task(self.slot_schedulers[slot_idx].schedule(ready_rows, slot_idx, self.dispatch))
            self.tasks.add(task)
            # TODO: is there a cheaper way to do this?
            task.add_done_callback(self.tasks.discard)
