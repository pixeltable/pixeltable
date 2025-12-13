import logging
import random

import sqlalchemy as sql
from rich.progress import Column, Progress, ProgressColumn, Task, TaskID, Text, TextColumn

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
    elapsed_time_task_id: TaskID | None
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
        self.elapsed_time_task_id = None
        self.progress_reporters = {}

        self.batch_size = batch_size
        self.profile = exprs.ExecProfile(row_builder)
        self.conn = None
        self.pk_clause = pk_clause
        self.ignore_errors = ignore_errors
        self.random_seed = random.randint(0, 1 << 63)

    def add_progress_reporter(self, desc: str, unit: str) -> ProgressReporter:
        """Records new ProgressReporter for the given desc/unit, or returns the existing one."""
        assert self.progress is not None
        key = f'{desc}_{unit}'
        if key in self.progress_reporters:
            return self.progress_reporters[key]
        reporter = ProgressReporter(self.progress, desc, unit)
        self.progress_reporters[key] = reporter
        return reporter

    def start_progress(self) -> None:
        """Create Progress object and start the timer. Idempotent."""
        if not self.show_progress or self.progress is not None:
            return

        def make_progress() -> Progress:
            return Progress(
                TextColumn('[progress.description]{task.description}', table_column=Column(min_width=40)),
                AdaptiveNumericColumn(),
                TextColumn('[progress.completed_1] {task.fields[unit_1]}', justify='left'),
                ' ',
                TextColumn('[progress.completed_2] {task.fields[unit_2]}', justify='left'),
                transient=True,  # remove after stop()
            )

        self.progress = Env.get().start_progress(make_progress)

    def stop_progress(self) -> None:
        """Stop the timer and print the final progress report. Idempotent."""
        if not self.show_progress or self.progress is None:
            return

        # self.progress.stop()
        # self.progress.console.clear()
        # for some reason, the progress bar is not cleared automatically in jupyter
        # if getattr(builtins, '__IPYTHON__', False):
        #     from IPython.display import clear_output
        #
        #     clear_output(wait=True)

        # report the total throughput of the last stage
        # last_reporter = next(reversed(self.progress_reporters.values()))
        # last_total = last_reporter.total
        # elapsed = time.monotonic() - self.progress_start
        # self.progress.console.print(
        #     f'{_print_number(last_total)} {last_reporter.unit} in {elapsed:.2f}s '
        #     f'({last_total / elapsed:.2f}{last_reporter.unit}/s)'
        # )
        #


def _print_number(val: float | int) -> str:
    if isinstance(val, int):
        return str(val)
    else:
        assert isinstance(val, float)
        if val < 1.0:
            return f'{val:.3f}'
        elif val < 10.0:
            return f'{val:.2f}'
        elif val < 100.0:
            return f'{val:.1f}'
        else:
            return f'{int(val)}'


class AdaptiveNumericColumn(ProgressColumn):
    """
    Custom column for adaptive float/int formatting.
    Renders completed count as an integer for whole numbers and as a float otherwise.
    """

    def render(self, task: Task) -> Text:
        formatted_value = _print_number(task.completed)
        return Text(formatted_value, style='progress.completed', justify='right')
