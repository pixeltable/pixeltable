from __future__ import annotations

import logging
from typing import List
from uuid import UUID

import sqlalchemy.orm as orm

from .table_version import TableVersion
from .table import Table
from .column import Column
from pixeltable.env import Env

_ID_RE = r'[a-zA-Z]\w*'
_PATH_RE = f'{_ID_RE}(\\.{_ID_RE})*'


_logger = logging.getLogger('pixeltable')

class View(Table):
    def __init__(self, dir_id: UUID, tbl_version: TableVersion):
        super().__init__(tbl_version.id, dir_id, tbl_version)

    @classmethod
    def display_name(cls) -> str:
        return 'view'

    @classmethod
    def create(
            cls, dir_id: UUID, name: str, base: TableVersion, cols: List[Column], predicate: 'exprs.Predicate',
            num_retained_versions: int) -> View:
        with orm.Session(Env.get().engine, future=True) as session:
            tbl_version = TableVersion.create(
                dir_id, name, cols, base, predicate, num_retained_versions, None, None, None, None, session)
            view = cls(dir_id, tbl_version)

            from pixeltable.plan import Planner
            plan, schema_col_info, idx_col_info, num_values_per_row = Planner.create_view_load_plan(tbl_version)
            num_rows, num_excs, cols_with_excs = \
                tbl_version.store_tbl.insert_rows(plan, schema_col_info, idx_col_info, session.connection())
            session.commit()
            _logger.info(f'created view {name}, id={tbl_version.id}')
            msg = f'created view {name} with {num_rows} rows, {num_excs} exceptions'
            print(msg)
            return view

