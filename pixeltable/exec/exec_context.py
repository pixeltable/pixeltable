from typing import Optional, List

import sqlalchemy as sql

import pixeltable.exprs as exprs

class ExecContext:
    """Class for execution runtime constants"""
    def __init__(
            self, row_builder: exprs.RowBuilder, *, show_pbar: bool = False, batch_size: int = 0,
            pk_clause: Optional[List[sql.ClauseElement]] = None, num_computed_exprs: int = 0,
            ignore_errors: bool = False
    ):
        self.show_pbar = show_pbar
        self.batch_size = batch_size
        self.profile = exprs.ExecProfile(row_builder)
        # num_rows is used to compute the total number of computed cells used for the progress bar
        self.num_rows: Optional[int] = None
        self.conn: Optional[sql.engine.Connection] = None  # if present, use this to execute SQL queries
        self.pk_clause = pk_clause
        self.num_computed_exprs = num_computed_exprs
        self.ignore_errors = ignore_errors
