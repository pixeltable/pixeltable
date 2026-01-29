import importlib.util
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

    title: str | None  # used in progress reporting
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
        show_progress: bool | None = None,
    ):
        self.title = None
        self.row_builder = row_builder
        if show_progress is not None:
            self.show_progress = show_progress
        else:
            self.show_progress = Env.get().verbosity >= 1 and Env.get().is_interactive()

        # disable progress reporting in Jupyter if ipywidgets is not installed
        if Env.get().is_notebook() and importlib.util.find_spec('ipywidgets') is None:
            self.show_progress = False

        self.progress = None
        self.progress_reporters = {}

        self.batch_size = batch_size
        self.profile = exprs.ExecProfile(row_builder)
        self.conn = None
        self.pk_clause = pk_clause
        self.ignore_errors = ignore_errors
        self.random_seed = random.randint(0, 1 << 63)

    def add_progress_reporter(self, desc: str, unit_1: str, unit_2: str | None = None) -> ProgressReporter | None:
        """Records new ProgressReporter for the given desc/units, or returns the existing one."""
        if not self.show_progress:
            return None
        assert self.progress is not None
        if desc in self.progress_reporters:
            return self.progress_reporters[desc]
        desc = desc if self.title is None else f'| {desc}'
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
                redirect_stdout=False,  # avoid bad tqdm interaction
                redirect_stderr=False,  # avoid bad tqdm interaction
            )

        self.progress = Env.get().start_progress(create_progress)
        if self.title is not None:
            self.progress.add_task(f'{self.title}:', total_1='', unit_1='', total_2='', unit_2='')
