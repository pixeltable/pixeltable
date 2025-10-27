import random

import sqlalchemy as sql

from pixeltable import exprs


class ExecContext:
    """Class for execution runtime constants"""

    row_builder: exprs.RowBuilder
    profile: exprs.ExecProfile
    show_pbar: bool
    batch_size: int
    num_rows: int | None
    conn: sql.engine.Connection | None
    pk_clause: list[sql.ClauseElement] | None
    num_computed_exprs: int
    ignore_errors: bool
    random_seed: int  # general-purpose source of randomness with execution scope

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        *,
        show_pbar: bool = False,
        batch_size: int = 0,
        pk_clause: list[sql.ClauseElement] | None = None,
        num_computed_exprs: int = 0,
        ignore_errors: bool = False,
    ):
        self.show_pbar = show_pbar
        self.batch_size = batch_size
        self.row_builder = row_builder
        self.profile = exprs.ExecProfile(row_builder)
        # num_rows is used to compute the total number of computed cells used for the progress bar
        self.num_rows = None
        self.conn = None  # if present, use this to execute SQL queries
        self.pk_clause = pk_clause
        self.num_computed_exprs = num_computed_exprs
        self.ignore_errors = ignore_errors
        self.random_seed = random.randint(0, 1 << 63)
