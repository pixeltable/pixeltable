from __future__ import annotations

import logging

from rich.progress import Progress, TaskID

_logger = logging.getLogger('pixeltable')


class ProgressReporter:
    """
    Represents a single Task, attached to a Progress instance.

    Task creation is deferred until the first update() call, in order to avoid useless output.
    """

    task_id: TaskID | None
    progress: Progress
    reports_bytes: bool  # if True, automatically scales the reported numbers to human-readable units
    desc: str
    unit_1: str
    total_1: float
    unit_2: str | None
    total_2: float | None

    def __init__(self, progress: Progress, desc: str, unit_1: str, unit_2: str | None):
        self.task_id = None
        self.progress = progress
        #self.reports_bytes = unit == 'B'
        self.desc = desc
        self.unit_1 = unit_1
        self.total_1 = 0
        self.unit_2 = unit_2
        self.total_2 = 0 if unit_2 is not None else None

    def _create_task(self) -> None:
        if self.task_id is None:
            self.task_id = self.progress.add_task(self.desc, unit_1=self.unit_1, unit_2 = '' if self.unit_2 is None else self.unit_2)

    def _get_display_unit(self) -> tuple[int, str]:
        # scale to human-readable unit
        scale: int
        unit: str
        if self.total < 2**10:
            scale = 0
            unit = 'B '
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

    def update(self, advance_1: float, advance_2: float | None = None) -> None:
        self._create_task()
        self.total_1 += advance_1
        if self.unit_2 is not None:
            assert advance_2 is not None
            self.total_2 += advance_2

        total = self.total
        unit = self.unit
        if self.reports_bytes:
            scale, unit = self._get_display_unit()
            total /= 2**scale
        self.progress.update(self.task_id, completed=total, unit=unit)

    def finalize(self) -> None:
        if self.task_id is None:
            # nothing to finalize
            return
        self._create_task()
        total = self.total
        unit = self.unit
        if self.reports_bytes:
            scale, unit = self._get_display_unit()
            total /= 2**scale
        self.progress.update(self.task_id, completed=total, unit=unit)
