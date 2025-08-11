from typing import Optional, Any

import sqlalchemy as sql
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from pixeltable import exprs


class ExecContext:
    """Class for execution runtime constants"""

    row_builder: exprs.RowBuilder
    show_pbar: bool
    progress: Optional[Progress]
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
        self.progress = None
        self.batch_size = batch_size
        self.profile = exprs.ExecProfile(row_builder)
        # self.num_rows: Optional[int] = None
        self.conn = None
        self.pk_clause = pk_clause
        # self.num_computed_exprs = num_computed_exprs
        self.ignore_errors = ignore_errors

    def add_pbar(self, desc: str, unit: str) -> Any:
        if self.progress is None:
            # self.progress = Progress(
            #     TextColumn("[progress.description]{task.description}"),
            #     BarColumn(),
            #     TextColumn("[progress.completed]{task.completed} {task.fields[unit]}"),
            #     TimeRemainingColumn(),
            # )
            self.progress = Progress(
                TextColumn('[progress.description]{task.description}'),
                BarColumn(),  # Display only the completed count and rate, ideal for streaming data
                TextColumn('[progress.completed]{task.completed}', justify='right'),
                '•',
                TextColumn('[progress.percentage]{task.fields[rate]}[/progress.percentage]', justify='right'),
                TimeElapsedColumn(),
            )
            self.progress.start()

        task_id = self.progress.add_task(desc, unit=unit, total=None, rate='0/s')
        return task_id

    def close_pbars(self) -> None:
        if self.progress is not None:
            self.progress.stop()
