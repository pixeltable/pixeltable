from __future__ import annotations

import glob
import hashlib
import logging
import os
import warnings
from collections import OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

import pixeltable.exceptions as excs
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
    keys_retrieved_this_session: set[str]  # keys retrieved (downloaded or accessed) this session
    keys_evicted_this_session: set[str]  # keys that were evicted after having been retrieved this session
    has_warned_about_eviction: bool

    ColumnStats = namedtuple('FileCacheColumnStats', ('tbl_id', 'col_id', 'num_files', 'total_size'))
    CacheStats = namedtuple(
        'FileCacheStats',
        ('total_size', 'num_requests', 'num_hits', 'num_evictions', 'column_stats')
    )

    @classmethod
    def get(cls) -> FileCache:
        if cls.__instance is None:
            cls.init()
        return cls.__instance

    @classmethod
    def init(cls) -> None:
        cls.__instance = cls()

    def __init__(self):
        self.cache = OrderedDict()
        self.total_size = 0
        self.capacity_bytes = Env.get()._cache_size_mb * (1 << 20)
        self.num_requests = 0
        self.num_hits = 0
        self.num_evictions = 0
        self.keys_retrieved_this_session = set()
        self.keys_evicted_this_session = set()
        self.has_warned_about_eviction = False
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
            # We need to store the entries to remove in a list, because we can't remove items from a dict while iterating
            entries_to_remove = list(self.cache.values())
            _logger.debug(f'clearing {self.num_files()} entries from file cache')
            self.num_requests, self.num_hits, self.num_evictions = 0, 0, 0
            self.keys_retrieved_this_session.clear()
            self.keys_evicted_this_session.clear()
            self.has_warned_about_eviction = False
        else:
            entries_to_remove = [e for e in self.cache.values() if e.tbl_id == tbl_id]
            _logger.debug(f'clearing {self.num_files(tbl_id)} entries from file cache for table {tbl_id}')
        for entry in entries_to_remove:
            os.remove(entry.path)
            del self.cache[entry.key]
            self.total_size -= entry.size

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
        entry.last_used = file_info.st_mtime
        self.cache.move_to_end(key, last=True)
        self.num_hits += 1
        self.keys_retrieved_this_session.add(key)
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
        if not self.has_warned_about_eviction and key in self.keys_evicted_this_session:
            warnings.warn(
                f'A media file was retrieved multiple times this session:\n'
                f'{url}\n'
                'It had to be downloaded a second time, because it was evicted from the file cache after its first access.\n'
                'Consider increasing the cache size; you can do this by setting the value of `cache_size_mb` in $PIXELTABLE_HOME/config.toml.',
                excs.PixeltableWarning
            )
            self.has_warned_about_eviction = True
        self.keys_retrieved_this_session.add(key)
        entry = CacheEntry(key, tbl_id, col_id, file_info.st_size, file_info.st_mtime, path.suffix)
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
            if lru_entry.key in self.keys_retrieved_this_session:
                # This key was retrieved at some point earlier this session and is now being evicted.
                # Make a record of the eviction, so that we can generate a warning later if the key is retrieved again.
                self.keys_evicted_this_session.add(lru_entry.key)
            os.remove(str(lru_entry.path))
            _logger.debug(f'evicted entry for cell {lru_entry.key} from file cache')

    def set_capacity(self, capacity_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes
        self.ensure_capacity(0)  # evict entries if necessary

    def stats(self) -> CacheStats:
        # collect column stats
        # (tbl_id, col_id) -> (num_files, total_size)
        d: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
        for entry in self.cache.values():
            t = d[(entry.tbl_id, entry.col_id)]
            t[0] += 1
            t[1] += entry.size
        col_stats = [
            self.ColumnStats(tbl_id, col_id, num_files, size) for (tbl_id, col_id), (num_files, size) in d.items()
        ]
        col_stats.sort(key=lambda e: e[3], reverse=True)
        return self.CacheStats(self.total_size, self.num_requests, self.num_hits, self.num_evictions, col_stats)

    def debug_print(self) -> None:
        for entry in self.cache.values():
            print(f'CacheEntry: tbl_id={entry.tbl_id}, col_id={entry.col_id}, size={entry.size}')
