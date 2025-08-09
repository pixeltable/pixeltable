import sys
from typing import Optional

import sqlalchemy as sql
from tqdm.auto import tqdm

from pixeltable import exprs


class ExecContext:
    """Class for execution runtime constants"""

    row_builder: exprs.RowBuilder
    show_pbar: bool
    pbars: list[tqdm]
    batch_size: int  # 0: no batching
    profile: exprs.ExecProfile
    conn: Optional[sql.engine.Connection]  # if present, use this to execute SQL queries
    pk_clause: Optional[list[sql.ClauseElement]]
    # num_computed_exprs: int  # number of exprs that need to be computed (ie, not materialized by a SqlNode)
    ignore_errors: bool

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        *,
        show_pbar: bool = False,
        batch_size: int = 0,
        pk_clause: Optional[list[sql.ClauseElement]] = None,
        num_computed_exprs: int = 0,
        ignore_errors: bool = False,
    ):
        self.row_builder = row_builder
        self.show_pbar = show_pbar
        self.pbars = []
        self.batch_size = batch_size
        self.profile = exprs.ExecProfile(row_builder)
        # self.num_rows: Optional[int] = None
        self.conn = None
        self.pk_clause = pk_clause
        # self.num_computed_exprs = num_computed_exprs
        self.ignore_errors = ignore_errors

    def add_pbar(self, desc: str, unit: str) -> tqdm:
        pbar = tqdm(desc=desc, unit=unit, file=sys.stderr, position=len(self.pbars), leave=False)
        self.pbars.append(pbar)
        return pbar

    def close_pbars(self) -> None:
        for pbar in self.pbars:
            pbar.close()
