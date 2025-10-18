from typing import TYPE_CHECKING, Optional

import sqlalchemy as sql

from pixeltable import exprs

if TYPE_CHECKING:
    from pixeltable.plan import SampleClause


class ExecContext:
    """Class for execution runtime constants"""

    row_builder: exprs.RowBuilder
    profile: exprs.ExecProfile
    show_pbar: bool
    batch_size: int
    num_rows: Optional[int]
    conn: Optional[sql.engine.Connection]
    pk_clause: Optional[list[sql.ClauseElement]]
    num_computed_exprs: int
    ignore_errors: bool
    sample_clause: Optional['SampleClause']

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
        self.sample_clause = None
