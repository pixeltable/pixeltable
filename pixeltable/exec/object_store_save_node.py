from __future__ import annotations

import dataclasses
import itertools
import logging
from collections import defaultdict, deque
from concurrent import futures
from pathlib import Path
from typing import AsyncIterator, Iterator, NamedTuple

from pixeltable import exprs
from pixeltable.utils.object_stores import ObjectOps, ObjectPath, StorageTarget
from pixeltable.utils.progress_reporter import ProgressReporter

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class ObjectStoreSaveNode(ExecNode):
    """Save files into designated object store(s).

    Each row may have multiple files that need to be saved to a destination.
    Each file may be referenced by more than one column in the row.
    Each file may have multiple destinations, e.g., S3 bucket and local file system.
    If there are multiple destinations, the file cannot be moved to any destination
    until it has been copied to all of the other destinations.
    Diagrammatically:
        Row -> [src_path1, src_path2, ...]
            src_path -> [dest1, dest2, ...]
                dest1: [row_location1, row_location2, ...]
    Paths with multiple destinations are removed from the TempStore only after all destination copies are complete.

    TODO:
    - Process a row at a time and limit the number of in-flight rows to control memory usage
    """

    QUEUE_DEPTH_HIGH_WATER = 50  # target number of in-flight requests
    QUEUE_DEPTH_LOW_WATER = 20  # target number of in-flight requests
    BATCH_SIZE = 16
    MAX_WORKERS = 15

    class WorkDesignator(NamedTuple):
        """Specify the source and destination for a WorkItem"""

        src_path: str  # source of the file to be processed
        destination: str  # destination URI for the file to be processed
        file_size: int  # in bytes

    class WorkItem(NamedTuple):
        src_path: Path
        destination: str | None
        info: exprs.ColumnSlotIdx  # column info for the file being processed
        destination_count: int = 1  # number of unique destinations for this file

    retain_input_order: bool  # if True, return rows in the exact order they were received
    file_col_info: list[exprs.ColumnSlotIdx]

    # execution state
    num_returned_rows: int

    # ready_rows: rows that are ready to be returned, ordered by row idx;
    # the implied row idx of ready_rows[0] is num_returned_rows
    ready_rows: deque[exprs.DataRow | None]

    in_flight_rows: dict[int, ObjectStoreSaveNode.RowState]  # rows with in-flight work; id(row) -> RowState
    in_flight_requests: dict[
        futures.Future, WorkDesignator
    ]  # in-flight requests to save paths: Future -> WorkDesignator
    in_flight_work: dict[
        WorkDesignator, list[tuple[exprs.DataRow, exprs.ColumnSlotIdx]]
    ]  # WorkDesignator -> [(row, info)]

    input_finished: bool
    row_idx: Iterator[int | None]

    # progress reporting
    progress_reporter: ProgressReporter | None

    @dataclasses.dataclass
    class RowState:
        row: exprs.DataRow
        idx: int | None  # position in input stream; None if we don't retain input order
        num_missing: int  # number of references to media files in this row
        delete_destinations: list[Path]  # paths to delete after all copies are complete

    def __init__(self, file_col_info: list[exprs.ColumnSlotIdx], input: ExecNode, retain_input_order: bool = True):
        # input_/output_exprs=[]: we don't have anything to evaluate
        super().__init__(input.row_builder, [], [], input)
        self.retain_input_order = retain_input_order
        self.file_col_info = file_col_info

        self.num_returned_rows = 0
        self.ready_rows = deque()
        self.in_flight_rows = {}
        self.in_flight_requests = {}
        self.in_flight_work = {}
        self.input_finished = False
        self.row_idx = itertools.count() if retain_input_order else itertools.repeat(None)
        self.progress_reporter = None
        assert self.QUEUE_DEPTH_HIGH_WATER > self.QUEUE_DEPTH_LOW_WATER

    @property
    def queued_work(self) -> int:
        return len(self.in_flight_requests)

    def _open(self) -> None:
        self.progress_reporter = self.ctx.add_progress_reporter('Uploads', 'objects', 'B')

    async def get_input_batch(self, input_iter: AsyncIterator[DataRowBatch]) -> DataRowBatch | None:
        """Get the next batch of input rows, or None if there are no more rows"""
        try:
            input_batch = await anext(input_iter)
            if input_batch is None:
                self.input_finished = True
            return input_batch
        except StopAsyncIteration:
            self.input_finished = True
            return None

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        input_iter = aiter(self.input)
        with futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            while True:
                # Create work to fill the queue to the high water mark ... ?without overrunning the in-flight row limit.
                while not self.input_finished and self.queued_work < self.QUEUE_DEPTH_HIGH_WATER:
                    input_batch = await self.get_input_batch(input_iter)
                    if input_batch is not None:
                        self.__process_input_batch(input_batch, executor)

                # Wait for enough completions to enable more queueing or if we're done
                while self.queued_work > self.QUEUE_DEPTH_LOW_WATER or (self.input_finished and self.queued_work > 0):
                    done, _ = futures.wait(self.in_flight_requests, return_when=futures.FIRST_COMPLETED)
                    self.__process_completions(done, ignore_errors=self.ctx.ignore_errors)

                # Emit results to meet batch size requirements or empty the in-flight row queue
                if self.__has_ready_batch() or (
                    len(self.ready_rows) > 0 and self.input_finished and self.queued_work == 0
                ):
                    # create DataRowBatch from the first BATCH_SIZE ready rows
                    batch = DataRowBatch(self.row_builder)
                    rows = [self.ready_rows.popleft() for _ in range(min(self.BATCH_SIZE, len(self.ready_rows)))]
                    for row in rows:
                        assert row is not None
                        batch.add_row(row)
                    self.num_returned_rows += len(rows)
                    _logger.debug(f'returning {len(rows)} rows')
                    yield batch

                if self.input_finished and self.queued_work == 0 and len(self.ready_rows) == 0:
                    return

    def __has_ready_batch(self) -> bool:
        """True if there are >= BATCH_SIZES entries in ready_rows and the first BATCH_SIZE ones are all non-None"""
        return (
            sum(int(row is not None) for row in itertools.islice(self.ready_rows, self.BATCH_SIZE)) == self.BATCH_SIZE
        )

    def __add_ready_row(self, row: exprs.DataRow, row_idx: int | None) -> None:
        if row_idx is None:
            self.ready_rows.append(row)
        else:
            # extend ready_rows to accommodate row_idx
            idx = row_idx - self.num_returned_rows
            if idx >= len(self.ready_rows):
                self.ready_rows.extend([None] * (idx - len(self.ready_rows) + 1))
            self.ready_rows[idx] = row

    def __process_completions(self, done: set[futures.Future], ignore_errors: bool) -> None:
        from pixeltable.utils.local_store import TempStore

        num_objects = 0
        num_bytes = 0

        for f in done:
            work_designator = self.in_flight_requests.pop(f)
            new_file_url, exc = f.result()
            if exc is not None and not ignore_errors:
                raise exc
            assert new_file_url is not None

            if exc is None:
                num_objects += 1
                num_bytes += work_designator.file_size

            # add the local path/exception to the slots that reference the url
            for row, info in self.in_flight_work.pop(work_designator):
                if exc is not None:
                    self.row_builder.set_exc(row, info.slot_idx, exc)
                else:
                    row.file_urls[info.slot_idx] = new_file_url

                state = self.in_flight_rows[id(row)]
                state.num_missing -= 1
                if state.num_missing == 0:
                    # All operations for this row are complete. Delete all files which had multiple destinations
                    for src_path in state.delete_destinations:
                        TempStore.delete_media_file(src_path)
                    del self.in_flight_rows[id(row)]
                    self.__add_ready_row(row, state.idx)

        if self.ctx.show_progress:
            self.progress_reporter.update(num_objects, num_bytes)

    def __process_input_row(self, row: exprs.DataRow) -> list[ObjectStoreSaveNode.WorkItem]:
        """Process a batch of input rows, generating a list of work"""
        from pixeltable.utils.local_store import LocalStore, TempStore

        # Create a list of work to do for media storage in this row
        row_idx = next(self.row_idx)
        row_to_do: list[ObjectStoreSaveNode.WorkItem] = []
        num_missing = 0
        unique_destinations: dict[Path, int] = defaultdict(int)  # destination -> count of unique destinations

        for info in self.file_col_info:
            col, index = info
            # we may need to store this imagehave yet to store this image
            if row.prepare_col_val_for_save(index, col):
                row.file_urls[index] = row.save_media_to_temp(index, col)

            url = row.file_urls[index]
            if url is None:
                # nothing to do
                continue

            assert row.excs[index] is None
            assert col.col_type.is_media_type()

            destination = info.col.destination
            if destination is not None:
                soa = ObjectPath.parse_object_storage_addr(destination, False)
                if soa.storage_target == StorageTarget.LOCAL_STORE and LocalStore(soa).resolve_url(url) is not None:
                    # A local non-default destination was specified, and the url already points there
                    continue

            src_path = LocalStore.file_url_to_path(url)
            if src_path is None:
                # The url does not point to a local file, do not attempt to copy/move it
                continue

            if destination is None and not TempStore.contains_path(src_path):
                # Do not copy local file URLs to the LocalStore
                continue

            work_designator = ObjectStoreSaveNode.WorkDesignator(str(src_path), destination, src_path.stat().st_size)
            locations = self.in_flight_work.get(work_designator)
            if locations is not None:
                # we've already seen this
                locations.append((row, info))
                num_missing += 1
                continue

            work_item = ObjectStoreSaveNode.WorkItem(src_path, destination, info)
            row_to_do.append(work_item)
            self.in_flight_work[work_designator] = [(row, info)]
            num_missing += 1
            unique_destinations[src_path] += 1

        # Update work items to reflect the number of unique destinations
        new_to_do = []
        for work_item in row_to_do:
            if unique_destinations[work_item.src_path] == 1 and TempStore.contains_path(work_item.src_path):
                new_to_do.append(work_item)
            else:
                new_to_do.append(
                    ObjectStoreSaveNode.WorkItem(
                        work_item.src_path,
                        work_item.destination,
                        work_item.info,
                        destination_count=unique_destinations[work_item.src_path] + 1,
                        # +1 for the TempStore destination
                    )
                )
        delete_destinations = [k for k, v in unique_destinations.items() if v > 1 and TempStore.contains_path(k)]
        row_to_do = new_to_do

        if len(row_to_do) > 0:
            self.in_flight_rows[id(row)] = self.RowState(
                row, row_idx, num_missing, delete_destinations=delete_destinations
            )
        else:
            self.__add_ready_row(row, row_idx)
        return row_to_do

    def __process_input_batch(self, input_batch: DataRowBatch, executor: futures.ThreadPoolExecutor) -> None:
        """Process a batch of input rows, submitting temporary files for upload"""
        work_to_do: list[ObjectStoreSaveNode.WorkItem] = []

        for row in input_batch:
            row_to_do = self.__process_input_row(row)
            if len(row_to_do) > 0:
                work_to_do.extend(row_to_do)

        for work_item in work_to_do:
            # determine size before file gets moved
            file_size = work_item.src_path.stat().st_size
            f = executor.submit(self.__persist_media_file, work_item)
            self.in_flight_requests[f] = ObjectStoreSaveNode.WorkDesignator(
                str(work_item.src_path), work_item.destination, file_size
            )
            _logger.debug(f'submitted {work_item}')

    def __persist_media_file(self, work_item: WorkItem) -> tuple[str | None, Exception | None]:
        """Move data from the TempStore to another location"""
        src_path = work_item.src_path
        col = work_item.info.col
        assert col.destination == work_item.destination
        try:
            new_file_url = ObjectOps.put_file(col, src_path, work_item.destination_count == 1)
            return new_file_url, None
        except Exception as e:
            _logger.debug(f'Failed to move/copy {src_path}: {e}', exc_info=e)
            return None, e
