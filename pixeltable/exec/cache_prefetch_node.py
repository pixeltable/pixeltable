from __future__ import annotations

import dataclasses
import itertools
import logging
import threading
import urllib.parse
import urllib.request
from collections import deque
from concurrent import futures
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional
from uuid import UUID

from pixeltable import catalog, env, exceptions as excs, exprs
from pixeltable.utils.filecache import FileCache

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class CachePrefetchNode(ExecNode):
    """Brings files with external URLs into the cache

    TODO:
    - adapting the number of download threads at runtime to maximize throughput
    """

    BATCH_SIZE = 16
    NUM_EXECUTOR_THREADS = 16

    retain_input_order: bool  # if True, return rows in the exact order they were received
    file_col_info: list[exprs.ColumnSlotIdx]
    boto_client: Optional[Any]
    boto_client_lock: threading.Lock

    # execution state
    batch_tbl_version: Optional[catalog.TableVersionHandle]  # needed to construct output batches
    num_returned_rows: int

    # ready_rows: rows that are ready to be returned, ordered by row idx;
    # the implied row idx of ready_rows[0] is num_returned_rows
    ready_rows: deque[Optional[exprs.DataRow]]

    in_flight_rows: dict[int, CachePrefetchNode.RowState]  # rows with in-flight urls; id(row) -> RowState
    in_flight_requests: dict[futures.Future, str]  # in-flight requests for urls; future -> URL
    in_flight_urls: dict[str, list[tuple[exprs.DataRow, exprs.ColumnSlotIdx]]]  # URL -> [(row, info)]
    input_finished: bool
    row_idx: Iterator[Optional[int]]

    @dataclasses.dataclass
    class RowState:
        row: exprs.DataRow
        idx: Optional[int]  # position in input stream; None if we don't retain input order
        num_missing: int  # number of missing URLs in this row

    def __init__(
        self, tbl_id: UUID, file_col_info: list[exprs.ColumnSlotIdx], input: ExecNode, retain_input_order: bool = True
    ):
        # input_/output_exprs=[]: we don't have anything to evaluate
        super().__init__(input.row_builder, [], [], input)
        self.retain_input_order = retain_input_order
        self.file_col_info = file_col_info

        # clients for specific services are constructed as needed, because it's time-consuming
        self.boto_client = None
        self.boto_client_lock = threading.Lock()

        self.batch_tbl_version = None
        self.num_returned_rows = 0
        self.ready_rows = deque()
        self.in_flight_rows = {}
        self.in_flight_requests = {}
        self.in_flight_urls = {}
        self.input_finished = False
        self.row_idx = itertools.count() if retain_input_order else itertools.repeat(None)

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        input_iter = self.input.__aiter__()
        with futures.ThreadPoolExecutor(max_workers=self.NUM_EXECUTOR_THREADS) as executor:
            # we create enough in-flight requests to fill the first batch
            while not self.input_finished and self.__num_pending_rows() < self.BATCH_SIZE:
                await self.__submit_input_batch(input_iter, executor)

            while True:
                # try to assemble a full batch of output rows
                if not self.__has_ready_batch() and len(self.in_flight_requests) > 0:
                    self.__wait_for_requests()

                # try to create enough in-flight requests to fill the next batch
                while not self.input_finished and self.__num_pending_rows() < self.BATCH_SIZE:
                    await self.__submit_input_batch(input_iter, executor)

                if len(self.ready_rows) > 0:
                    # create DataRowBatch from the first BATCH_SIZE ready rows
                    batch = DataRowBatch(self.batch_tbl_version, self.row_builder)
                    rows = [self.ready_rows.popleft() for _ in range(min(self.BATCH_SIZE, len(self.ready_rows)))]
                    for row in rows:
                        assert row is not None
                        batch.add_row(row)
                    self.num_returned_rows += len(rows)
                    _logger.debug(f'returning {len(rows)} rows')
                    yield batch

                if self.input_finished and self.__num_pending_rows() == 0:
                    return

    def __num_pending_rows(self) -> int:
        return len(self.in_flight_rows) + len(self.ready_rows)

    def __has_ready_batch(self) -> bool:
        """True if there are >= BATCH_SIZES entries in ready_rows and the first BATCH_SIZE ones are all non-None"""
        return (
            sum(int(row is not None) for row in itertools.islice(self.ready_rows, self.BATCH_SIZE)) == self.BATCH_SIZE
        )

    def __ready_prefix_len(self) -> int:
        """Length of the non-None prefix of ready_rows (= what we can return right now)"""
        return sum(1 for _ in itertools.takewhile(lambda x: x is not None, self.ready_rows))

    def __add_ready_row(self, row: exprs.DataRow, row_idx: Optional[int]) -> None:
        if row_idx is None:
            self.ready_rows.append(row)
        else:
            # extend ready_rows to accommodate row_idx
            idx = row_idx - self.num_returned_rows
            if idx >= len(self.ready_rows):
                self.ready_rows.extend([None] * (idx - len(self.ready_rows) + 1))
            self.ready_rows[idx] = row

    def __wait_for_requests(self) -> None:
        """Wait for in-flight requests to complete until we have a full batch of rows"""
        file_cache = FileCache.get()
        _logger.debug(f'waiting for requests; ready_batch_size={self.__ready_prefix_len()}')
        while not self.__has_ready_batch() and len(self.in_flight_requests) > 0:
            done, _ = futures.wait(self.in_flight_requests, return_when=futures.FIRST_COMPLETED)
            for f in done:
                url = self.in_flight_requests.pop(f)
                tmp_path, exc = f.result()
                local_path: Optional[Path] = None
                if tmp_path is not None:
                    # register the file with the cache for the first column in which it's missing
                    assert url in self.in_flight_urls
                    _, info = self.in_flight_urls[url][0]
                    local_path = file_cache.add(info.col.tbl.id, info.col.id, url, tmp_path)
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
                        _logger.debug(f'row {state.idx} is ready (ready_batch_size={self.__ready_prefix_len()})')

    async def __submit_input_batch(
        self, input: AsyncIterator[DataRowBatch], executor: futures.ThreadPoolExecutor
    ) -> None:
        assert not self.input_finished
        input_batch: Optional[DataRowBatch]
        try:
            input_batch = await anext(input)
        except StopAsyncIteration:
            input_batch = None
        if input_batch is None:
            self.input_finished = True
            return
        if self.batch_tbl_version is None:
            self.batch_tbl_version = input_batch.tbl

        file_cache = FileCache.get()

        # URLs from this input batch that aren't already in the file cache;
        # we use a list to make sure we submit urls in the order in which they appear in the output, which minimizes
        # the time it takes to get the next batch together
        cache_misses: list[str] = []

        url_pos: dict[str, int] = {}  # url -> row_idx; used for logging
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

    def __fetch_url(self, url: str) -> tuple[Optional[Path], Optional[Exception]]:
        """Fetches a remote URL into Env.tmp_dir and returns its path"""
        _logger.debug(f'fetching url={url} thread_name={threading.current_thread().name}')
        parsed = urllib.parse.urlparse(url)
        # Use len(parsed.scheme) > 1 here to ensure we're not being passed
        # a Windows filename
        assert len(parsed.scheme) > 1 and parsed.scheme != 'file'
        # preserve the file extension, if there is one
        extension = ''
        if parsed.path:
            p = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed.path)))
            extension = p.suffix
        tmp_path = env.Env.get().create_tmp_path(extension=extension)
        try:
            _logger.debug(f'Downloading {url} to {tmp_path}')
            if parsed.scheme == 's3':
                from pixeltable.utils.s3 import get_client

                with self.boto_client_lock:
                    if self.boto_client is None:
                        config = {
                            'max_pool_connections': self.NUM_EXECUTOR_THREADS + 4,  # +4: leave some headroom
                            'connect_timeout': 5,
                            'read_timeout': 30,
                            'retries': {'max_attempts': 3, 'mode': 'adaptive'},
                        }
                        self.boto_client = get_client(**config)
                self.boto_client.download_file(parsed.netloc, parsed.path.lstrip('/'), str(tmp_path))
            elif parsed.scheme in ('http', 'https'):
                with urllib.request.urlopen(url) as resp, open(tmp_path, 'wb') as f:
                    data = resp.read()
                    f.write(data)
            else:
                raise AssertionError(f'Unsupported URL scheme: {parsed.scheme}')
            _logger.debug(f'Downloaded {url} to {tmp_path}')
            return tmp_path, None
        except Exception as e:
            # we want to add the file url to the exception message
            exc = excs.Error(f'Failed to download {url}: {e}')
            _logger.debug(f'Failed to download {url}: {e}', exc_info=e)
            if not self.ctx.ignore_errors:
                raise exc from None  # suppress original exception
            return None, exc
