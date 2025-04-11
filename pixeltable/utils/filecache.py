from __future__ import annotations

import glob
import hashlib
import logging
import os
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional
from uuid import UUID

import pixeltable.exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env

_logger = logging.getLogger('pixeltable')


@dataclass
class CacheEntry:
    key: str
    tbl_id: UUID
    col_id: int
    size: int
    last_used: datetime
    ext: str

    @property
    def path(self) -> Path:
        return Env.get().file_cache_dir / f'{self.tbl_id.hex}_{self.col_id}_{self.key}{self.ext}'

    @classmethod
    def from_file(cls, path: Path) -> CacheEntry:
        components = path.stem.split('_')
        assert len(components) == 3
        tbl_id = UUID(components[0])
        col_id = int(components[1])
        key = components[2]
        file_info = os.stat(str(path))
        # We use the last modified time (file_info.st_mtime) as the timestamp; `FileCache` will touch the file
        # each time it is retrieved, so that the mtime of the file will always represent the last used time of
        # the cache entry.
        last_used = datetime.fromtimestamp(file_info.st_mtime, tz=timezone.utc)
        return cls(key, tbl_id, col_id, file_info.st_size, last_used, path.suffix)


class FileCache:
    """
    A local cache of external (eg, S3) file references in cells of a stored table (ie, table or view).

    Cache entries are identified by a hash of the file url and stored in Env.filecache_dir. The time of last
    access of a cache entries is its file's mtime.

    TODO:
    - implement MRU eviction for queries that exceed the capacity
    """

    __instance: Optional[FileCache] = None

    cache: OrderedDict[str, CacheEntry]
    total_size: int
    capacity_bytes: int
    num_requests: int
    num_hits: int
    num_evictions: int
    keys_retrieved: set[str]  # keys retrieved (downloaded or accessed) this session
    keys_evicted_after_retrieval: set[str]  # keys that were evicted after having been retrieved this session

    # A key is added to this set when it is already present in `keys_evicted_this_session` and is downloaded again.
    # In other words, for a key to be added to this set, the following sequence of events must occur in this order:
    # - It is retrieved during this session (either because it was newly downloaded, or because it was in the cache
    #   at the start of the session and was accessed at some point during the session)
    # - It is subsequently evicted
    # - It is subsequently retrieved a second time ("download after a previous retrieval")
    # The contents of this set will be used to generate a more informative warning.
    evicted_working_set_keys: set[str]
    new_redownload_witnessed: bool  # whether a new re-download has occurred since the last time a warning was issued

    class FileCacheColumnStats(NamedTuple):
        tbl_id: UUID
        col_id: int
        num_files: int
        total_size: int

    class FileCacheStats(NamedTuple):
        total_size: int
        num_requests: int
        num_hits: int
        num_evictions: int
        column_stats: list[FileCache.FileCacheColumnStats]

    @classmethod
    def get(cls) -> FileCache:
        if cls.__instance is None:
            cls.init()
        return cls.__instance

    @classmethod
    def init(cls) -> None:
        cls.__instance = cls()

    def __init__(self) -> None:
        self.cache = OrderedDict()
        self.total_size = 0
        self.capacity_bytes = int(Env.get()._file_cache_size_g * (1 << 30))
        self.num_requests = 0
        self.num_hits = 0
        self.num_evictions = 0
        self.keys_retrieved = set()
        self.keys_evicted_after_retrieval = set()
        self.evicted_working_set_keys = set()
        self.new_redownload_witnessed = False
        paths = glob.glob(str(Env.get().file_cache_dir / '*'))
        entries = [CacheEntry.from_file(Path(path_str)) for path_str in paths]
        # we need to insert entries in access order
        entries.sort(key=lambda e: e.last_used)
        for entry in entries:
            self.cache[entry.key] = entry
            self.total_size += entry.size

    def avg_file_size(self) -> int:
        if len(self.cache) == 0:
            return 0
        return int(self.total_size / len(self.cache))

    def num_files(self, tbl_id: Optional[UUID] = None) -> int:
        if tbl_id is None:
            return len(self.cache)
        return sum(e.tbl_id == tbl_id for e in self.cache.values())

    def clear(self, tbl_id: Optional[UUID] = None) -> None:
        """
        For testing purposes: allow resetting capacity and stats.
        """
        if tbl_id is None:
            # We need to store the entries to remove in a list, because we can't remove items from a dict
            # while iterating
            entries_to_remove = list(self.cache.values())
            _logger.debug(f'clearing {self.num_files()} entries from file cache')
            self.num_requests, self.num_hits, self.num_evictions = 0, 0, 0
            self.keys_retrieved.clear()
            self.keys_evicted_after_retrieval.clear()
            self.new_redownload_witnessed = False
        else:
            entries_to_remove = [e for e in self.cache.values() if e.tbl_id == tbl_id]
            _logger.debug(f'clearing {self.num_files(tbl_id)} entries from file cache for table {tbl_id}')
        for entry in entries_to_remove:
            os.remove(entry.path)
            del self.cache[entry.key]
            self.total_size -= entry.size

    def emit_eviction_warnings(self) -> None:
        if self.new_redownload_witnessed:
            # Compute the additional capacity that would be needed in order to retain all the re-downloaded files
            extra_capacity_needed = sum(self.cache[key].size for key in self.evicted_working_set_keys)
            suggested_cache_size = self.capacity_bytes + extra_capacity_needed + (1 << 30)
            warnings.warn(
                f'{len(self.evicted_working_set_keys)} media file(s) had to be downloaded multiple times this session, '
                'because they were evicted\nfrom the file cache after their first access. The total size '
                f'of the evicted file(s) is {round(extra_capacity_needed / (1 << 30), 1)} GiB.\n'
                f'Consider increasing the cache size to at least {round(suggested_cache_size / (1 << 30), 1)} GiB '
                f'(it is currently {round(self.capacity_bytes / (1 << 30), 1)} GiB).\n'
                f'You can do this by setting the value of `file_cache_size_g` in: {Config.get().config_file}',
                excs.PixeltableWarning,
                stacklevel=2,
            )
            self.new_redownload_witnessed = False

    def _url_hash(self, url: str) -> str:
        h = hashlib.sha256()
        h.update(url.encode())
        return h.hexdigest()

    def lookup(self, url: str) -> Optional[Path]:
        self.num_requests += 1
        key = self._url_hash(url)
        entry = self.cache.get(key, None)
        if entry is None:
            _logger.debug(f'file cache miss for {url}')
            return None
        # update mtime and cache
        path = entry.path
        path.touch(exist_ok=True)
        file_info = os.stat(str(path))
        entry.last_used = datetime.fromtimestamp(file_info.st_mtime)
        self.cache.move_to_end(key, last=True)
        self.num_hits += 1
        self.keys_retrieved.add(key)
        _logger.debug(f'file cache hit for {url}')
        return path

    def add(self, tbl_id: UUID, col_id: int, url: str, path: Path) -> Path:
        """Adds url at 'path' to cache and returns its new path.
        'path' will not be accessible after this call. Retains the extension of 'path'.
        """
        file_info = os.stat(str(path))
        self.ensure_capacity(file_info.st_size)
        key = self._url_hash(url)
        assert key not in self.cache
        if key in self.keys_evicted_after_retrieval:
            # This key was evicted after being retrieved earlier this session, and is now being retrieved again.
            # Add it to `keys_multiply_downloaded` so that we may generate a warning later.
            self.evicted_working_set_keys.add(key)
            self.new_redownload_witnessed = True
        self.keys_retrieved.add(key)
        entry = CacheEntry(
            key, tbl_id, col_id, file_info.st_size, datetime.fromtimestamp(file_info.st_mtime), path.suffix
        )
        self.cache[key] = entry
        self.total_size += entry.size
        new_path = entry.path
        os.rename(str(path), str(new_path))
        new_path.touch(exist_ok=True)
        _logger.debug(f'added entry for cell {url} to file cache')
        return new_path

    def ensure_capacity(self, size: int) -> None:
        """
        Evict entries from the cache until there is at least 'size' bytes of free space.
        """
        while len(self.cache) > 0 and self.total_size + size > self.capacity_bytes:
            _, lru_entry = self.cache.popitem(last=False)
            self.total_size -= lru_entry.size
            self.num_evictions += 1
            if lru_entry.key in self.keys_retrieved:
                # This key was retrieved at some point earlier this session and is now being evicted.
                # Make a record of the eviction, so that we can generate a warning later if the key is retrieved again.
                self.keys_evicted_after_retrieval.add(lru_entry.key)
            os.remove(str(lru_entry.path))
            _logger.debug(
                f'evicted entry for cell {lru_entry.key} from file cache (of size {lru_entry.size // (1 << 20)} MiB)'
            )

    def set_capacity(self, capacity_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes
        self.ensure_capacity(0)  # evict entries if necessary

    def stats(self) -> FileCacheStats:
        # collect column stats
        # (tbl_id, col_id) -> (num_files, total_size)
        d: dict[tuple[UUID, int], list[int]] = defaultdict(lambda: [0, 0])
        for entry in self.cache.values():
            t = d[entry.tbl_id, entry.col_id]
            t[0] += 1
            t[1] += entry.size
        col_stats = [
            self.FileCacheColumnStats(tbl_id, col_id, num_files, size)
            for (tbl_id, col_id), (num_files, size) in d.items()
        ]
        col_stats.sort(key=lambda e: e[3], reverse=True)
        return self.FileCacheStats(self.total_size, self.num_requests, self.num_hits, self.num_evictions, col_stats)

    def debug_print(self) -> None:
        for entry in self.cache.values():
            _logger.debug(f'CacheEntry: tbl_id={entry.tbl_id}, col_id={entry.col_id}, size={entry.size}')
