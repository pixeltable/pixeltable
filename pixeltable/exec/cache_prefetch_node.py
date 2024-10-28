from __future__ import annotations

import concurrent.futures
import logging
import threading
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Any, Tuple, Dict
from uuid import UUID

import pixeltable.env as env
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
from pixeltable.utils.filecache import FileCache
from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')

class CachePrefetchNode(ExecNode):
    """Brings files with external URLs into the cache

    TODO:
    - maintain a queue of row batches, in order to overlap download and evaluation
    - adapting the number of download threads at runtime to maximize throughput
    """
    def __init__(self, tbl_id: UUID, file_col_info: List[exprs.ColumnSlotIdx], input: ExecNode):
        # []: we don't have anything to evaluate
        super().__init__(input.row_builder, [], [], input)
        self.tbl_id = tbl_id
        self.file_col_info = file_col_info

        # clients for specific services are constructed as needed, because it's time-consuming
        self.boto_client: Optional[Any] = None
        self.boto_client_lock = threading.Lock()

    def __next__(self) -> DataRowBatch:
        input_batch = next(self.input)

        # collect external URLs that aren't already cached, and set DataRow.file_paths for those that are
        file_cache = FileCache.get()
        cache_misses: List[Tuple[exprs.DataRow, exprs.ColumnSlotIdx]] = []
        missing_url_rows: Dict[str, List[exprs.DataRow]] = defaultdict(list)  # URL -> rows in which it's missing
        for row in input_batch:
            for info in self.file_col_info:
                url = row.file_urls[info.slot_idx]
                if url is None or row.file_paths[info.slot_idx] is not None:
                    # nothing to do
                    continue
                if url in missing_url_rows:
                    missing_url_rows[url].append(row)
                    continue
                local_path = file_cache.lookup(url)
                if local_path is None:
                    cache_misses.append((row, info))
                    missing_url_rows[url].append(row)
                else:
                    row.set_file_path(info.slot_idx, str(local_path))

        # download the cache misses in parallel
        # TODO: set max_workers to maximize throughput
        futures: Dict[concurrent.futures.Future, Tuple[exprs.DataRow, exprs.ColumnSlotIdx]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for row, info in cache_misses:
                futures[executor.submit(self._fetch_url, row, info.slot_idx)] = (row, info)
            for future in concurrent.futures.as_completed(futures):
                # TODO:  does this need to deal with recoverable errors (such as retry after throttling)?
                tmp_path = future.result()
                if tmp_path is None:
                    continue
                row, info = futures[future]
                url = row.file_urls[info.slot_idx]
                local_path = file_cache.add(self.tbl_id, info.col.id, url, tmp_path)
                _logger.debug(f'PrefetchNode: cached {url} as {local_path}')
                for row in missing_url_rows[url]:
                    row.set_file_path(info.slot_idx, str(local_path))

        return input_batch

    def _fetch_url(self, row: exprs.DataRow, slot_idx: int) -> Optional[str]:
        """Fetches a remote URL into Env.tmp_dir and returns its path"""
        url = row.file_urls[slot_idx]
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
            return tmp_path
        except Exception as e:
            # we want to add the file url to the exception message
            exc = excs.Error(f'Failed to download {url}: {e}')
            self.row_builder.set_exc(row, slot_idx, exc)
            if not self.ctx.ignore_errors:
                raise exc from None  # suppress original exception
        return None

