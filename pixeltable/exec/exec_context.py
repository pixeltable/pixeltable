import logging
import random

import sqlalchemy as sql
from rich.progress import Column, Progress, TextColumn

from pixeltable import exprs
from pixeltable.env import Env
from pixeltable.utils.progress_reporter import ProgressReporter

_logger = logging.getLogger('pixeltable')


class ExecContext:
    """Class for execution runtime constants"""

    row_builder: exprs.RowBuilder
    show_progress: bool
    progress: Progress | None
    progress_reporters: dict[str, ProgressReporter]
    batch_size: int  # 0: no batching
    profile: exprs.ExecProfile
    conn: sql.engine.Connection | None  # if present, use this to execute SQL queries
    pk_clause: list[sql.ClauseElement] | None
    ignore_errors: bool
    random_seed: int  # general-purpose source of randomness with execution scope

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        *,
        batch_size: int = 0,
        pk_clause: list[sql.ClauseElement] | None = None,
        ignore_errors: bool = False,
    ):
        self.row_builder = row_builder
        self.show_progress = Env.get().verbosity >= 1 and Env.get().is_interactive()
        self.progress = None
        self.progress_reporters = {}

        self.batch_size = batch_size
        self.profile = exprs.ExecProfile(row_builder)
        self.conn = None
        self.pk_clause = pk_clause
        self.ignore_errors = ignore_errors
        self.random_seed = random.randint(0, 1 << 63)

    def add_progress_reporter(self, desc: str, unit_1: str, unit_2: str | None = None) -> ProgressReporter:
        """Records new ProgressReporter for the given desc/units, or returns the existing one."""
        assert self.progress is not None
        if desc in self.progress_reporters:
            return self.progress_reporters[desc]
        reporter = ProgressReporter(self.progress, desc, unit_1, unit_2)
        self.progress_reporters[desc] = reporter
        return reporter

    def start_progress(self) -> None:
        """Create Progress object and start the timer. Idempotent."""
        if not self.show_progress or self.progress is not None:
            return

        def create_progress() -> Progress:
            return Progress(
                TextColumn('[progress.description]{task.description}', table_column=Column(min_width=40)),
                TextColumn('{task.fields[total_1]}', justify='right', table_column=Column(min_width=10)),
                TextColumn('{task.fields[unit_1]}', justify='left'),
                TextColumn('{task.fields[total_2]}', justify='right', table_column=Column(min_width=10)),
                TextColumn('{task.fields[unit_2]}', justify='left'),
                transient=True,  # remove at end
                #transient=False,  # Keep visible, clear manually at end
            )

        self.progress = Env.get().start_progress(create_progress)
