import logging
import random
import time
from typing import Optional

import sqlalchemy as sql
from rich.progress import Progress, ProgressColumn, Task, TaskID, Text, TextColumn

from pixeltable import exprs

_logger = logging.getLogger('pixeltable')


class ExecContext:
    """Class for execution runtime constants"""

    row_builder: exprs.RowBuilder
    show_progress: bool
    progress: Optional[Progress]
    progress_start: float  # time.monotonic() of progress.start()
    progress_reporters: dict[str, 'ExecContext.ProgressReporter']
    elapsed_time_task_id: Optional[TaskID]
    batch_size: int  # 0: no batching
    profile: exprs.ExecProfile
    conn: Optional[sql.engine.Connection]  # if present, use this to execute SQL queries
    pk_clause: Optional[list[sql.ClauseElement]]
    ignore_errors: bool
    random_seed: int  # general-purpose source of randomness with execution scope

    class ProgressReporter:
        """
        Represents a single Task, attached to ExecCtx.progress.

        Task creation is deferred until the first update() call, in order to avoid useless output.
        """

        task_id: Optional[TaskID]
        ctx: 'ExecContext'
        last_update_ts: Optional[float]
        reports_bytes: bool  # if True, automatically scales the reported numbers to human-readable units
        total: float
        desc: str
        unit: str

        def __init__(self, ctx: 'ExecContext', desc: str, unit: str):
            self.ctx = ctx
            self.desc = desc
            self.unit = unit
            self.reports_bytes = unit == 'B'
            self.task_id = None
            self.last_update_ts = None
            self.total = 0

        def _create_task(self) -> None:
            if self.task_id is None:
                self.task_id = self.ctx.progress.add_task(self.desc, rate='0/s', unit=self.unit)
                self.last_update_ts = time.monotonic()  # start now

        def _get_display_unit(self) -> tuple[int, str]:
            # scale to human-readable unit
            scale: int
            unit: str
            if self.total < 2**10:
                scale = 0
                unit = 'B'
            elif self.total < 2**20:
                scale = 10
                unit = 'KB'
            elif self.total < 2**30:
                scale = 20
                unit = 'MB'
            else:
                scale = 30
                unit = 'GB'
            return scale, unit

        def update(self, advance: float) -> None:
            _logger.debug(f'ProgressReporter.update({self.desc}): advance={advance}')
            self._create_task()
            now = time.monotonic()
            self.total += advance

            time_delta = now - self.last_update_ts
            rate = advance / time_delta if time_delta > 0 else 0.0
            total = self.total
            unit = self.unit
            if self.reports_bytes:
                scale, unit = self._get_display_unit()
                rate /= 2**scale
                total /= 2**scale
            self.last_update_ts = now
            self.ctx.progress.update(self.task_id, completed=total, rate=f'{rate:.2f} {unit}/s', unit=unit)
            # elapsed = now - self.ctx.progress_start
            # self.ctx.progress.update(self.ctx.elapsed_time_task_id, completed=elapsed, rate='')

        def finalize(self) -> None:
            if self.last_update_ts is None:
                # nothing to finalize
                return
            self._create_task()
            # show aggregate rate since start
            elapsed = time.monotonic() - self.ctx.progress_start
            rate = self.total / elapsed if elapsed > 0 else 0.0
            total = self.total
            unit = self.unit
            if self.reports_bytes:
                scale, unit = self._get_display_unit()
                rate /= 2**scale
                total /= 2**scale
            self.ctx.progress.update(self.task_id, completed=total, unit=unit, rate=f'{rate:.2f} {unit}/s')

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        *,
        show_progress: bool = False,
        batch_size: int = 0,
        pk_clause: Optional[list[sql.ClauseElement]] = None,
        ignore_errors: bool = False,
    ):
        self.row_builder = row_builder
        self.show_progress = show_progress
        self.progress = None
        self.elapsed_time_task_id = None
        self.progress_reporters = {}

        self.batch_size = batch_size
        self.profile = exprs.ExecProfile(row_builder)
        self.conn = None
        self.pk_clause = pk_clause
        self.ignore_errors = ignore_errors
        self.random_seed = random.randint(0, 1 << 63)

    def add_progress_reporter(self, desc: str, unit: str) -> 'ExecContext.ProgressReporter':
        """Records new ProgressReporter for the given desc/unit, or returns the existing one."""
        assert self.progress is not None
        key = f'{desc}_{unit}'
        if key in self.progress_reporters:
            return self.progress_reporters[key]
        reporter = self.ProgressReporter(self, desc, unit)
        self.progress_reporters[key] = reporter
        return reporter

    def start_progress(self) -> None:
        """Create Progress object and start the timer. Idempotent."""
        if not self.show_progress or self.progress is not None:
            return
        self.progress = Progress(
            TextColumn('[progress.description]{task.description}'),
            AdaptiveNumericColumn(),
            TextColumn('[progress.completed] {task.fields[unit]}', justify='left'),
            ' ',
            TextColumn('[progress.percentage]{task.fields[rate]}[/progress.percentage]', justify='right'),
        )
        # self.elapsed_time_task_id = self.progress.add_task('Total time', unit='s', rate='')
        self.progress.start()
        self.progress_start = time.monotonic()

    def stop_progress(self) -> None:
        """Stop the timer and print the final progress report. Idempotent."""
        if not self.show_progress or self.progress is None:
            return
        for reporter in self.progress_reporters.values():
            reporter.finalize()
        self.progress.refresh()
        self.progress.stop()


class AdaptiveNumericColumn(ProgressColumn):
    """
    Custom column for adaptive float/int formatting.
    Renders completed count as an integer for whole numbers and as a float otherwise.
    """

    def render(self, task: Task) -> Text:
        formatted_value: str
        if isinstance(task.completed, int):
            formatted_value = str(task.completed)
        else:
            assert isinstance(task.completed, float)
            if task.completed < 1.0:
                formatted_value = f'{task.completed:.3f}'
            elif task.completed < 10.0:
                formatted_value = f'{task.completed:.2f}'
            elif task.completed < 100.0:
                formatted_value = f'{task.completed:.1f}'
            else:
                formatted_value = f'{int(task.completed)}'

        return Text(formatted_value, style='progress.completed', justify='right')
