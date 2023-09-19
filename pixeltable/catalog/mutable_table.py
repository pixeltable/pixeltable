from __future__ import annotations

import logging
from typing import Optional, List
from uuid import UUID

import sqlalchemy.orm as orm

from .column import Column
from .table_version import TableVersion
from .table import Table
from pixeltable.env import Env

_ID_RE = r'[a-zA-Z]\w*'
_PATH_RE = f'{_ID_RE}(\\.{_ID_RE})*'


_logger = logging.getLogger('pixeltable')

class MutableTable(Table):
    """A :py:class:`Table` that can be modified.
    """
    def __init__(self, dir_id: UUID, tbl_version: TableVersion):
        super().__init__(tbl_version.id, dir_id, tbl_version)

    @classmethod
    def display_name(cls) -> str:
        return 'table'

    # MODULE-LOCAL, NOT PUBLIC
    @classmethod
    def create(
            cls, dir_id: UUID, name: str, cols: List[Column],
            num_retained_versions: int,
            extract_frames_from: Optional[str], extracted_frame_col: Optional[str],
            extracted_frame_idx_col: Optional[str], extracted_fps: Optional[int],
    ) -> MutableTable:
        with orm.Session(Env.get().engine, future=True) as session:
            tbl_version = TableVersion.create(
                dir_id, name, cols, None, None, num_retained_versions, extract_frames_from, extracted_frame_col,
                extracted_frame_idx_col, extracted_fps, session)
            tbl = cls(dir_id, tbl_version)
            session.commit()
            _logger.info(f'created table {name}, id={tbl_version.id}')
            return tbl

