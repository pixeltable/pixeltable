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
from typing import Optional, Any, Iterator
from uuid import UUID

import pixeltable.env as env
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
from pixeltable import catalog
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

    retain_input_order: bool
    file_col_info: list[exprs.ColumnSlotIdx]
    boto_client: Optional[Any]
    boto_client_lock: threading.Lock

    # execution state
    batch_tbl_version: Optional[catalog.TableVersion]  # needed to construct output batches
    num_returned_rows: int
    ready_rows: deque[Optional[exprs.DataRow]]  # the implied row idx of ready_rows[0] is num_returned_rows
    in_flight_rows: dict[int, CachePrefetchNode.RowState]  # id(row) -> RowState
    in_flight_requests: dict[futures.Future, str]  # future -> URL
    in_flight_urls: dict[str, list[tuple[exprs.DataRow, exprs.ColumnSlotIdx]]]  # URL -> [(row, info)]
    input_finished: bool
    row_idx: Iterator[Optional[int]]

    @dataclasses.dataclass
    class RowState:
        row: exprs.DataRow
        idx: Optional[int]  # position in input stream; None if we don't retain input order
        num_missing: int  # number of missing URLs in this row

    def __init__(
            self, tbl_id: UUID, file_col_info: list[exprs.ColumnSlotIdx], input: ExecNode,
            retain_input_order: bool = True):
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

    def __iter__(self) -> Iterator[DataRowBatch]:
        input_iter = iter(self.input)
        with futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix='prefetch') as executor:
            # we create enough in-flight requests to fill the first batch
            while not self.input_finished and self._num_pending_rows() < self.BATCH_SIZE:
                self._submit_input_batch(input_iter, executor)

            while True:
                # try to assemble a full batch of output rows
                if not self._has_ready_batch() and len(self.in_flight_requests) > 0:
                    self._wait_for_requests()

                # try to create enough in-flight requests to fill the next batch
                while not self.input_finished and self._num_pending_rows() < self.BATCH_SIZE:
                    self._submit_input_batch(input_iter, executor)

                if len(self.ready_rows) > 0:
                    # create DataRowBatch from the first BATCH_SIZE ready rows
                    batch = DataRowBatch(self.batch_tbl_version, self.row_builder)
                    rows = [self.ready_rows.popleft() for _ in range(min(self.BATCH_SIZE, len(self.ready_rows)))]
                    for row in rows:
                        assert row is not None
                        batch.add_row(row)
                    self.num_returned_rows += len(rows)
                    yield batch

                if self.input_finished and self._num_pending_rows() == 0:
                    return

    def _num_pending_rows(self) -> int:
        return len(self.in_flight_rows) + len(self.ready_rows)

    def _has_ready_batch(self) -> bool:
        """True if the first BATCH_SIZE entries in ready_rows are all non-None"""
        return (
            sum(1 for row in itertools.islice(self.ready_rows, self.BATCH_SIZE) if row is not None) == self.BATCH_SIZE
        )

    def _add_ready_row(self, row: exprs.DataRow, row_idx: Optional[int]) -> None:
        if row_idx is None:
            self.ready_rows.append(row)
        else:
            # extend ready_rows to accommodate row_idx
            idx = row_idx - self.num_returned_rows
            if idx >= len(self.ready_rows):
                self.ready_rows.extend([None] * (idx - len(self.ready_rows) + 1))
            self.ready_rows[idx] = row

    def _wait_for_requests(self) -> None:
        """Wait for in-flight requests to complete until we have a full batch of rows"""
        file_cache = FileCache.get()
        while not self._has_ready_batch() and len(self.in_flight_requests) > 0:
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
                    _logger.debug(f'CachePrefetchNode: cached {url} as {local_path}')

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
                        self._add_ready_row(row, state.idx)
                        _logger.debug(f'CachePrefetchNode: row {state.idx} is ready')

    def _submit_input_batch(self, input: Iterator[DataRowBatch], executor: futures.ThreadPoolExecutor) -> None:
        assert not self.input_finished
        input_batch = next(input, None)
        if input_batch is None:
            self.input_finished = True
            return
        if self.batch_tbl_version is None:
            self.batch_tbl_version = input_batch.tbl

        # identify missing local files in input batch, or fill in their paths if they're already cached
        file_cache = FileCache.get()
        cache_misses: set[str] = set()  # URLs from this input batch that aren't already in the file cache
        for row in input_batch:
            num_missing = 0
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
                    cache_misses.add(url)
                    self.in_flight_urls[url] = [(row, info)]
                    num_missing += 1
                else:
                    row.set_file_path(info.slot_idx, str(local_path))

            row_idx = next(self.row_idx)
            if num_missing > 0:
                self.in_flight_rows[id(row)] = self.RowState(row, row_idx, num_missing)
            else:
                self._add_ready_row(row, row_idx)

        for url in cache_misses:
            f = executor.submit(self._fetch_url, url)
            _logger.debug(f'CachePrefetchNode: submitted {url}')
            self.in_flight_requests[f] = url

    def _fetch_url(self, url: str) -> tuple[Optional[Path], Optional[Exception]]:
        """Fetches a remote URL into Env.tmp_dir and returns its path"""
        _logger.debug(f'url={url} thread_name={threading.current_thread().name}')
        parsed = urllib.parse.urlparse(url)
        # Use len(parsed.scheme) > 1 here to ensure we're not being passed
        # a Windows filename
        assert len(parsed.scheme) > 1 and parsed.scheme != 'file'
        # preserve the file extension, if there is one
        extension = ''
        if parsed.path != '':
            p = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed.path)))
            extension = p.suffix
        tmp_path = env.Env.get().create_tmp_path(extension=extension)
        try:
            _logger.debug(f'Downloading {url} to {tmp_path}')
            if parsed.scheme == 's3':
                from pixeltable.utils.s3 import get_client
                with self.boto_client_lock:
                    if self.boto_client is None:
                        self.boto_client = get_client()
                    self.boto_client.download_file(parsed.netloc, parsed.path.lstrip('/'), str(tmp_path))
            elif parsed.scheme == 'http' or parsed.scheme == 'https':
                with urllib.request.urlopen(url) as resp, open(tmp_path, 'wb') as f:
                    data = resp.read()
                    f.write(data)
            else:
                assert False, f'Unsupported URL scheme: {parsed.scheme}'
            _logger.debug(f'Downloaded {url} to {tmp_path}')
            return tmp_path, None
        except Exception as e:
            # we want to add the file url to the exception message
            exc = excs.Error(f'Failed to download {url}: {e}')
            _logger.debug(f'Failed to download {url}: {e}', exc_info=e)
            if not self.ctx.ignore_errors:
                raise exc from None  # suppress original exception
            return None, exc
