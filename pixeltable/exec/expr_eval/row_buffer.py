from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from pixeltable import exprs

_logger = logging.getLogger('pixeltable')


class RowBuffer:
    """Fixed-length circular buffer of DataRows; knows how to maintain input order"""

    size: int
    row_pos_map: Optional[dict[int, int]]  # id(row) -> position of row in output; None if not maintaining order
    num_rows: int  # number of rows in the buffer
    num_ready: int  # number of consecutive non-None rows at head
    buffer: np.ndarray  # of object
    head_idx: int  # index of beginning of the buffer
    head_pos: int  # row position of the beginning of the buffer

    def __init__(self, size: int):
        self.size = size
        self.row_pos_map = None
        self.num_rows = 0
        self.num_ready = 0
        self.buffer = np.full(size, None, dtype=object)
        self.head_pos = 0
        self.head_idx = 0

    def set_row_pos_map(self, row_pos_map: dict[int, int]) -> None:
        self.row_pos_map = row_pos_map

    def add_row(self, row: exprs.DataRow) -> None:
        offset: int  # of new row from head
        if self.row_pos_map is not None:
            pos = self.row_pos_map.get(id(row))
            assert pos is not None and (pos - self.head_pos < self.size), f'{pos} {self.head_pos} {self.size}'
            offset = pos - self.head_pos
        else:
            offset = self.num_rows
        idx = (self.head_idx + offset) % self.size
        assert self.buffer[idx] is None

        self.buffer[idx] = row
        self.num_rows += 1
        if self.row_pos_map is not None:
            if offset == self.num_ready:
                # we have new ready rows; find out how many
                while offset < self.size and self.buffer[(self.head_idx + offset) % self.size] is not None:
                    offset += 1
                self.num_ready = offset
        else:
            self.num_ready += 1

    def get_rows(self, n: int) -> list[exprs.DataRow]:
        """Get up to n ready rows from head"""
        n = min(n, self.num_ready)
        if n == 0:
            return []
        rows: list[exprs.DataRow]
        if self.head_idx + n <= self.size:
            rows = self.buffer[self.head_idx : self.head_idx + n].tolist()
            self.buffer[self.head_idx : self.head_idx + n] = None
        else:
            rows = np.concatenate([self.buffer[self.head_idx :], self.buffer[: self.head_idx + n - self.size]]).tolist()
            self.buffer[self.head_idx :] = None
            self.buffer[: self.head_idx + n - self.size] = None
        self.head_pos += n
        self.head_idx = (self.head_idx + n) % self.size
        self.num_rows -= n
        self.num_ready -= n
        return rows
