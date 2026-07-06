from __future__ import annotations

import glob
import hashlib
import logging
import os
import re
import threading
import time
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple
from uuid import UUID

import pixeltable.exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.utils.file_lock import FileLock

_logger = logging.getLogger(__name__)

# <tbl_id.hex>_<col_id>_<url_hash><ext>
_CACHE_ENTRY_FILE_RE = re.compile(r'(?P<tbl_id>[0-9a-f]{32})_(?P<col_id>[0-9]+)_(?P<key>[0-9a-f]{64})(\..*)?')


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
        m = _CACHE_ENTRY_FILE_RE.fullmatch(path.name)
        if m is None:
            raise ValueError(f'not a cache entry file name: {path.name!r}')
        tbl_id = UUID(m.group('tbl_id'))
        col_id = int(m.group('col_id'))
        key = m.group('key')
        file_info = os.stat(str(path))
        # We use the last modified time (file_info.st_mtime) as the timestamp; `FileCache` will touch the file
        # each time it is retrieved, so that the mtime of the file will always represent the last used time of
        # the cache entry.
        last_used = datetime.fromtimestamp(file_info.st_mtime, tz=timezone.utc)
        return cls(key, tbl_id, col_id, file_info.st_size, last_used, path.suffix)


class FileCache:
    """
    A local cache of external (eg, S3) file references in cells of a stored table (ie, table or view).

    Cache entries are identified by a hash of the file url and stored in Env.file_cache_dir. The time of last
    access of a cache entry is its file's mtime.

    Concurrency:
    - The cache directory is shared state: multiple threads in this process, and multiple processes sharing the
      same file_cache_dir, operate on it concurrently.
    - The filesystem is the source of truth. self.cache is an in-process index that mirrors what this process
      has seen but is treated as advisory (every access needs to verify that it reflects the filesystem state and
      possibly correct the cached state).
    - A file is protected from eviction while it is in use, via a lease enforced through the file's mtime:
      lookup()/add() touch the file, and eviction skips any file whose mtime is within lease_seconds. The lease
      auto-expires, so a crashed holder never leaks a permanent pin.
    - Cross-process capacity enforcement is best-effort: two processes adding concurrently can exceed capacity_bytes,
      corrected as each subsequently evicts.

    TODO:
    - implement MRU eviction for queries that exceed the capacity
    """

    __instance: FileCache | None = None

    _lock: threading.RLock
    cache: OrderedDict[str, CacheEntry]

    # Serializes lease renewal (shared) against eviction's check-and-remove (exclusive) across processes, so
    # a renewal cannot land between ensure_capacity's mtime read and its os.remove. Held only for those brief
    # operations, never across a read (the lease covers the read). Its file is ignored by the dir scan (the
    # stem has no '_'-separated id/col/key parts).
    _evict_lock: FileLock

    # best-effort capacity tracking
    total_size: int
    capacity_bytes: int

    lease_seconds: float

    # stats collected for the current session
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
        if cls.__instance is not None:
            cls.__instance._evict_lock.close()
        cls.__instance = cls()

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # the lock file is ignored by the dir scan (doesn't match the scan regex).
        self._evict_lock = FileLock(Env.get().file_cache_dir / '.filecache.lock')

        self.total_size = 0
        self.capacity_bytes = int(Env.get()._file_cache_size_g * (1 << 30))
        self.lease_seconds = Env.get()._file_cache_lease_s
        self.num_requests = 0
        self.num_hits = 0
        self.num_evictions = 0
        self.keys_retrieved = set()
        self.keys_evicted_after_retrieval = set()
        self.evicted_working_set_keys = set()
        self.new_redownload_witnessed = False
        self._init_index()

    def _init_index(self) -> None:
        """Initialize the index from the directory's current contents."""
        entries: list[CacheEntry] = []
        for path_str in glob.glob(str(Env.get().file_cache_dir / '*')):
            path = Path(path_str)
            if _CACHE_ENTRY_FILE_RE.fullmatch(path.name) is None:
                # not a cache entry file
                continue
            try:
                entries.append(CacheEntry.from_file(path))
            except (ValueError, OSError):
                # the file matched the name pattern but vanished between the glob and the stat (another process
                # evicted it), or is otherwise unreadable; ignore it
                continue

        # a full rebuild: reset both so the scan replaces prior state instead of adding to it
        self.cache = OrderedDict()
        self.total_size = 0
        # we need to insert entries in access order
        entries.sort(key=lambda e: e.last_used)
        for entry in entries:
            self.cache[entry.key] = entry
            self.total_size += entry.size

    def avg_file_size(self) -> int:
        with self._lock:
            if len(self.cache) == 0:
                return 0
            return int(self.total_size / len(self.cache))

    def num_files(self, tbl_id: UUID | None = None) -> int:
        with self._lock:
            if tbl_id is None:
                return len(self.cache)
            return sum(e.tbl_id == tbl_id for e in self.cache.values())

    def clear(self, tbl_id: UUID | None = None) -> None:
        """
        For testing purposes: allow resetting capacity and stats.
        """
        with self._lock:
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
                self._remove_file(entry.path)
                self.cache.pop(entry.key, None)
                self.total_size -= entry.size

    def validate(self) -> None:
        """
        Validation: every entry tracked in self.cache still exists on disk.

        The reverse direction (every file on disk is tracked here) is intentionally not checked: the directory
        is shared, so it may legitimately contain files added by other processes that this index never saw.
        """
        with self._lock:
            for entry in self.cache.values():
                assert entry.path.exists(), f'{entry.path} does not exist'

    def emit_eviction_warnings(self) -> None:
        with self._lock:
            if self.new_redownload_witnessed:
                # Compute the additional capacity that would be needed in order to retain all the re-downloaded files
                extra_capacity_needed = sum(
                    self.cache[key].size for key in self.evicted_working_set_keys if key in self.cache
                )
                suggested_cache_size = self.capacity_bytes + extra_capacity_needed + (1 << 30)
                warnings.warn(
                    f'{len(self.evicted_working_set_keys)} media file(s) had to be downloaded multiple times '
                    'this session, because they were evicted\nfrom the file cache after their first access. The '
                    f'total size of the evicted file(s) is {round(extra_capacity_needed / (1 << 30), 1)} GiB.\n'
                    f'Consider increasing the cache size to at least {round(suggested_cache_size / (1 << 30), 1)} '
                    f'GiB (it is currently {round(self.capacity_bytes / (1 << 30), 1)} GiB).\n'
                    f'You can do this by setting the value of `file_cache_size_g` in: {Config.get().config_file}',
                    excs.PixeltableWarning,
                    stacklevel=2,
                )
                self.new_redownload_witnessed = False

    def _url_hash(self, url: str) -> str:
        h = hashlib.sha256()
        h.update(url.encode())
        return h.hexdigest()

    def _remove_file(self, path: Path) -> None:
        """Remove a cache file, if it still exists."""
        try:
            os.remove(str(path))
        except FileNotFoundError:
            pass

    def lookup(self, url: str) -> Path | None:
        """Look up a file in the cache by URL and return its Path, or None if not found.
        Updates the file's last used time, so it's safe to use for lease_seconds."""
        with self._lock:
            self.num_requests += 1
            key = self._url_hash(url)
            entry = self.cache.get(key, None)
            if entry is None:
                _logger.debug(f'file cache miss for {url}')
                return None

            path = entry.path
            try:
                with self._evict_lock.shared():
                    # renewing the lease must be atomic w.r.t. eviction's check-and-remove; os.utime also
                    # validates existence in one syscall, raising FileNotFoundError if already evicted
                    os.utime(str(path))
                    entry.last_used = datetime.fromtimestamp(os.stat(str(path)).st_mtime)
            except FileNotFoundError:
                # reconcile the index and report a miss
                del self.cache[key]
                self.total_size -= entry.size
                _logger.debug(f'file cache miss for {url}')
                return None

            self.cache.move_to_end(key, last=True)
            self.num_hits += 1
            self.keys_retrieved.add(key)
            _logger.debug(f'file cache hit for {url}')
            return path

    def add(self, tbl_id: UUID, col_id: int, url: str, path: Path) -> Path:
        """Adds url at 'path' to cache and returns its new path.
        'path' will not be accessible after this call. Retains the extension of 'path'.
        """
        with self._lock:
            key = self._url_hash(url)
            existing = self.cache.get(key, None)
            if existing is not None:
                try:
                    with self._evict_lock.shared():
                        # already cached (this process added it, or another thread/process won a concurrent
                        # download); renew the lease atomically w.r.t. eviction. os.utime raises FileNotFoundError
                        # if the file was removed since we created the self.cache entry.
                        os.utime(str(existing.path))
                        existing.last_used = datetime.fromtimestamp(os.stat(str(existing.path)).st_mtime)
                    # discard the redundant download and return the existing file
                    self._remove_file(path)
                    self.cache.move_to_end(key, last=True)
                    self.keys_retrieved.add(key)
                    return existing.path
                except FileNotFoundError:
                    # stale index entry: the file was evicted by another process; drop it and re-add below
                    del self.cache[key]
                    self.total_size -= existing.size

            file_info = os.stat(str(path))
            self.ensure_capacity(file_info.st_size)
            if key in self.keys_evicted_after_retrieval:
                # This key was evicted after being retrieved earlier this session, and is now being retrieved again.
                # Add it to `evicted_working_set_keys` so that we may generate a warning later.
                self.evicted_working_set_keys.add(key)
                self.new_redownload_witnessed = True
            self.keys_retrieved.add(key)
            entry = CacheEntry(
                key, tbl_id, col_id, file_info.st_size, datetime.fromtimestamp(file_info.st_mtime), path.suffix
            )
            new_path = entry.path
            # os.replace is atomic and overwrites a copy another process may have created for the same url
            os.replace(str(path), str(new_path))
            new_path.touch(exist_ok=True)
            self.cache[key] = entry
            self.total_size += entry.size
            _logger.debug(f'FileCache: cached url {url} with file name {new_path}')
            return new_path

    def ensure_capacity(self, size: int) -> None:
        """
        Evict entries until there is at least 'size' bytes of free space, without evicting leased (in-use) files:
        - if 'size' bytes cannot be freed because every remaining entry is leased (in concurrent use), raises
          FILE_CACHE_FULL.
        - size == 0 (best-effort shrink) doesn't raise.
        """
        if self.total_size + size <= self.capacity_bytes:
            return

        with self._lock:
            self._init_index()  # make sure we see the current state of the cache
            for key in list(self.cache.keys()):  # oldest-accessed first
                if self.total_size + size <= self.capacity_bytes:
                    return
                entry = self.cache[key]
                with self._evict_lock.exclusive():
                    # hold the lock across the mtime check and the removal, so a concurrent lease renewal cannot
                    # slip in between them and get its just-renewed file deleted
                    try:
                        mtime = os.stat(str(entry.path)).st_mtime
                    except OSError:
                        mtime = None  # already gone; fall through and reconcile the index
                    if mtime is not None and time.time() - mtime < self.lease_seconds:
                        # leased (in use here or in another process); skip it
                        continue
                    self._remove_file(entry.path)
                del self.cache[key]
                self.total_size -= entry.size
                self.num_evictions += 1
                if entry.key in self.keys_retrieved:
                    # This key was retrieved earlier this session and is now being evicted. Record the eviction, so
                    # that we can generate a warning later if the key is retrieved again.
                    self.keys_evicted_after_retrieval.add(entry.key)
                _logger.debug(
                    f'evicted entry for cell {entry.key} from file cache (of size {entry.size // (1 << 20)} MiB)'
                )

            if size > 0 and self.total_size + size > self.capacity_bytes:
                # every remaining entry is leased; we could not free enough space for the new file
                raise excs.RequestError(
                    excs.ErrorCode.FILE_CACHE_FULL,
                    f'The file cache ({self.capacity_bytes / (1 << 30):.2f} GB) is too small for the set of media '
                    f'files\nin concurrent use. Increase `file_cache_size_g` in {Config.get().config_file}.',
                )

    def set_capacity(self, capacity_bytes: int) -> None:
        with self._lock:
            self.capacity_bytes = capacity_bytes
            self.ensure_capacity(0)  # evict entries if necessary

    def set_lease_seconds(self, lease_seconds: float) -> None:
        with self._lock:
            self.lease_seconds = lease_seconds

    def stats(self) -> FileCacheStats:
        with self._lock:
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
        with self._lock:
            for entry in self.cache.values():
                _logger.debug(f'CacheEntry: tbl_id={entry.tbl_id}, col_id={entry.col_id}, size={entry.size}')
