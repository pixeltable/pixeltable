from typing import Optional, List, Tuple, Dict
from collections import OrderedDict, defaultdict
import os
import re
import glob
from dataclasses import dataclass
from pathlib import Path

from pixeltable.env import Env

@dataclass(eq=True, frozen=True)
class CellId:
    tbl_id: int  # store.Table.id
    col_id: int  # store.ColumnHistory.col_id
    row_id: int
    v_min: int


class CacheEntry:
    filename_pattern = re.compile(r'(\d+)_(\d+)_(\d+)_(\d+)\.(\w+)')

    def __init__(self, cell_id: CellId, size: int, last_accessed_ts: int, suffix: str):
        self.cell_id = cell_id
        self.size = size
        self.last_accessed_ts = last_accessed_ts
        self.suffix = suffix

    def filename(self) -> str:
        return f'{self.cell_id.tbl_id}_{self.cell_id.col_id}_{self.cell_id.v_min}_{self.cell_id.row_id}{self.suffix}'

    def path(self) -> Path:
        return Env.get().filecache_dir / self.filename()

    @classmethod
    def from_file(cls, path: Path) -> 'CacheEntry':
        matched = re.match(cls.filename_pattern, path.name)
        assert matched is not None
        tbl_id = int(matched.group(1))
        col_id = int(matched.group(2))
        row_id = int(matched.group(3))
        v_min = int(matched.group(4))
        suffix = f'.{matched.group(5)}'
        file_info = os.stat(str(path))
        return cls(CellId(tbl_id, col_id, row_id, v_min), file_info.st_size, file_info.st_mtime, suffix)


class FileCache:
    _instance: Optional['FileCache'] = None

    @classmethod
    def get(cls) -> 'FileCache':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        paths = glob.glob(str(Env.get().filecache_dir / '*'))
        self.cache: OrderedDict[CellId, CacheEntry] = OrderedDict()  # ordered by entry.last_accessed_ts
        self.total_size = 0
        entries = [CacheEntry.from_file(Path(path_str)) for path_str in paths]
        # we need to insert entries in order of last_accessed_ts
        entries.sort(key=lambda e: e.last_accessed_ts)
        for entry in entries:
            self.cache[entry.cell_id] = entry
            self.total_size += entry.size

    def num_files(self, tbl_id: Optional[int] = None) -> int:
        if tbl_id is None:
            return len(self.cache)
        entries = [e for e in self.cache.values() if e.cell_id.tbl_id == tbl_id]
        return len(entries)

    def clear(self, tbl_id: Optional[int] = None) -> None:
        entries = list(self.cache.values())  # list(): avoid dealing with values() return type
        if tbl_id is not None:
            entries = [e for e in entries if e.cell_id.tbl_id == tbl_id]
        for entry in entries:
            del self.cache[entry.cell_id]
            self.total_size -= entry.size
            os.remove(entry.path())

    def lookup(self, tbl_id: int, col_id: int, row_id: int, v_min: int) -> Optional[Path]:
        cell_id = CellId(tbl_id, col_id, row_id, v_min)
        entry = self.cache.get(cell_id, None)
        if entry is None:
            return None
        # update mtime and cache
        path = entry.path()
        path.touch(exist_ok=True)
        self.cache.move_to_end(cell_id, last=False)
        return path

    def add(self, tbl_id: int, col_id: int, row_id: int, v_min: int, query_ts: int, path: Path) -> None:
        """
        Possibly adds 'path' to cache. If it does, 'path' is not accessible afterwards.
        """
        file_info = os.stat(str(path))
        if self.total_size + file_info.st_size > Env.get().max_filecache_size:
            # evict entries until we're below the limit or until we run into entries the current query brought in
            while True:
                lru_entry = self.cache.popitem(last=True)
                if lru_entry.last_accessed_ts >= query_ts:
                    # the current query brought this entry in: switch to MRU and ignore this put()
                    self.cache[lru_entry.cell_id] = lru_entry  # need to put it back
                    return
                self.total_size -= lru_entry.size
                os.remove(str(lru_entry.path()))
                if self.total_size + file_info.st_size <= Env.get().max_filecache_size:
                    break

        cell_id = CellId(tbl_id, col_id, row_id, v_min)
        entry = CacheEntry(cell_id, file_info.st_size, file_info.st_mtime, path.suffix)
        self.cache[entry.cell_id] = entry
        self.total_size += entry.size
        os.rename(str(path), str(entry.path()))

    def stats(self) -> List[Tuple[int, int, int, int]]:
        """
        Returns list of (tbl-id, col-id, #files, size), in decreasing order of size.
        """
        d: Dict[Tuple[int, int], List[int]] = defaultdict(lambda: [0, 0])
        for entry in self.cache.values():
            t = d[(entry.cell_id.tbl_id, entry.cell_id.col_id)]
            t[0] += 1
            t[1] += entry.size
        result = [(tbl_id, col_id, num_files, size) for (tbl_id, col_id), (num_files, size) in d.items()]
        result.sort(key=lambda e: e[3], reverse=True)
        return result

    def debug_print(self) -> None:
        for entry in self.cache.values():
            print(f'CacheEntry: tbl_id={entry.cell_id.tbl_id}, col_id={entry.cell_id.col_id}, size={entry.size}')
