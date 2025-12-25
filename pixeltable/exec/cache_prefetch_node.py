from __future__ import annotations

import dataclasses
import itertools
import logging
from collections import deque
from concurrent import futures
from pathlib import Path
from typing import AsyncIterator, Iterator
from uuid import UUID

from pixeltable import exceptions as excs, exprs
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.http import fetch_url
from pixeltable.utils.progress_reporter import ProgressReporter

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class CachePrefetchNode(ExecNode):
    """Brings files with external URLs into the cache

    TODO:
    - Process a row at a time and limit the number of in-flight rows to control memory usage
    - Create asyncio.Tasks to consume our input in order to increase concurrency.
    """

    QUEUE_DEPTH_HIGH_WATER = 50  # target number of in-flight requests
    QUEUE_DEPTH_LOW_WATER = 20  # target number of in-flight requests
    BATCH_SIZE = 16
    MAX_WORKERS = 15

    retain_input_order: bool  # if True, return rows in the exact order they were received
    file_col_info: list[exprs.ColumnSlotIdx]

    # execution state
    num_returned_rows: int
    progress_reporter: ProgressReporter | None

    # ready_rows: rows that are ready to be returned, ordered by row idx;
    # the implied row idx of ready_rows[0] is num_returned_rows
    ready_rows: deque[exprs.DataRow | None]

    in_flight_rows: dict[int, CachePrefetchNode.RowState]  # rows with in-flight urls; id(row) -> RowState
    in_flight_requests: dict[futures.Future, str]  # in-flight requests for urls; future -> URL
    in_flight_urls: dict[str, list[tuple[exprs.DataRow, exprs.ColumnSlotIdx]]]  # URL -> [(row, info)]
    input_finished: bool
    row_idx: Iterator[int | None]

    @dataclasses.dataclass
    class RowState:
        row: exprs.DataRow
        idx: int | None  # position in input stream; None if we don't retain input order
        num_missing: int  # number of missing URLs in this row

    def __init__(
        self, tbl_id: UUID, file_col_info: list[exprs.ColumnSlotIdx], input: ExecNode, retain_input_order: bool = True
    ):
        # input_/output_exprs=[]: we don't have anything to evaluate
        super().__init__(input.row_builder, [], [], input)
        self.retain_input_order = retain_input_order
        self.file_col_info = file_col_info

        self.num_returned_rows = 0
        self.progress_reporter = None
        self.ready_rows = deque()
        self.in_flight_rows = {}
        self.in_flight_requests = {}
        self.in_flight_urls = {}
        self.input_finished = False
        self.row_idx = itertools.count() if retain_input_order else itertools.repeat(None)
        assert self.QUEUE_DEPTH_HIGH_WATER > self.QUEUE_DEPTH_LOW_WATER

    @property
    def queued_work(self) -> int:
        return len(self.in_flight_requests)

    def _open(self) -> None:
        self.progress_reporter = self.ctx.add_progress_reporter('Downloads', 'objects', 'B')

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
        file_cache = FileCache.get()
        num_objects = 0
        num_bytes = 0

        for f in done:
            url = self.in_flight_requests.pop(f)
            tmp_path, exc = f.result()
            if exc is not None and not ignore_errors:
                raise exc
            local_path: Path | None = None
            if tmp_path is not None:
                num_objects += 1
                num_bytes += tmp_path.stat().st_size

                # register the file with the cache for the first column in which it's missing
                assert url in self.in_flight_urls
                _, info = self.in_flight_urls[url][0]
                local_path = file_cache.add(info.col.get_tbl().id, info.col.id, url, tmp_path)
                _logger.debug(f'cached {url} as {local_path}')

            # add the local path/exception to the slots that reference the url
            for row, info in self.in_flight_urls.pop(url):
                if exc is not None:
                    self.row_builder.set_exc(row, info.slot_idx, exc)
                else:
                    assert local_path is not None
                    row.set_file_path(info.slot_idx, str(local_path))
                state = self.in_flight_rows[id(row)]
                state.num_missing -= 1
                if state.num_missing == 0:
                    del self.in_flight_rows[id(row)]
                    self.__add_ready_row(row, state.idx)

        if self.ctx.show_progress:
            self.progress_reporter.update(num_objects, num_bytes)

    def __process_input_batch(self, input_batch: DataRowBatch, executor: futures.ThreadPoolExecutor) -> None:
        """Process a batch of input rows, submitting URLs for download and adding ready rows to ready_rows"""
        file_cache = FileCache.get()

        # URLs from this input batch that aren't already in the file cache;
        # we use a list to make sure we submit urls in the order in which they appear in the output, which minimizes
        # the time it takes to get the next batch together
        cache_misses: list[str] = []

        url_pos: dict[str, int | None] = {}  # url -> row_idx; used for logging
        for row in input_batch:
            # identify missing local files in input batch, or fill in their paths if they're already cached
            num_missing = 0
            row_idx = next(self.row_idx)

            for info in self.file_col_info:
                url = row.file_urls[info.slot_idx]
                if url is None or row.file_paths[info.slot_idx] is not None:
                    # nothing to do
                    continue
                locations = self.in_flight_urls.get(url)
                if locations is not None:
                    # we've already seen this
                    locations.append((row, info))
                    num_missing += 1
                    continue

                local_path = file_cache.lookup(url)
                if local_path is None:
                    cache_misses.append(url)
                    self.in_flight_urls[url] = [(row, info)]
                    num_missing += 1
                    if url not in url_pos:
                        url_pos[url] = row_idx
                else:
                    row.set_file_path(info.slot_idx, str(local_path))

            if num_missing > 0:
                self.in_flight_rows[id(row)] = self.RowState(row, row_idx, num_missing)
            else:
                self.__add_ready_row(row, row_idx)

        _logger.debug(f'submitting {len(cache_misses)} urls')
        for url in cache_misses:
            f = executor.submit(self.__fetch_url, url)
            _logger.debug(f'submitted {url} for idx {url_pos[url]}')
            self.in_flight_requests[f] = url

    def __fetch_url(self, url: str) -> tuple[Path | None, Exception | None]:
        try:
            return fetch_url(url), None
        except Exception as e:
            # we want to add the file url to the exception message
            exc = excs.Error(f'Failed to download {url}: {e}')
            _logger.debug(f'Failed to download {url}: {e}', exc_info=e)
            return None, exc
