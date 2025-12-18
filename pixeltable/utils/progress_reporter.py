from __future__ import annotations

import logging
from typing import cast

from rich.progress import Progress, TaskID

_logger = logging.getLogger('pixeltable')


def _print_number(val: float | int | None) -> str:
    """Format a number adaptively for display."""
    if val is None:
        return ''
    if isinstance(val, int):
        return str(val)
    if val < 1.0:
        return f'{val:.3f}'
    elif val < 10.0:
        return f'{val:.2f} '  # trailing space for alignment
    elif val < 100.0:
        return f'{val:.1f}  '  # trailing spaces for alignment
    else:
        return f'{round(val)}    '  # trailing spaces for alignment


class ProgressReporter:
    """
    Represents a single Task, attached to a Progress instance.

    Task creation is deferred until the first update() call, in order to avoid useless output.
    Supports one or two value/unit pairs per task line.

    Progress updates:
    - Rich needs ipywidgets in a Jupyter notebook in order to display live Progress updates
    - trying to get this to work with a combination of Console(force_terminal=True, force_jupyter=False),
      Progress.refresh() and IPython.display.clear_output() didn't work
    """

    task_id: TaskID | None
    progress: Progress
    desc: str
    unit_1: str
    total_1: float | int
    unit_2: str | None
    total_2: float | int | None

    def __init__(self, progress: Progress, desc: str, unit_1: str, unit_2: str | None):
        self.task_id = None
        self.progress = progress
        self.desc = desc
        self.unit_1 = unit_1
        self.total_1 = 0
        self.unit_2 = unit_2
        self.total_2 = 0 if unit_2 is not None else None

    def _create_task(self) -> None:
        if self.task_id is None:
            self.task_id = self.progress.add_task(
                self.desc, total_1='', unit_1=self.unit_1, total_2='', unit_2=self.unit_2 if self.unit_2 else ''
            )

    def _scale_bytes(self, num_bytes: int) -> tuple[float, str]:
        # scales number of bytes to a human-readable unit; returns (scaled value, unit)
        if num_bytes < 2**10:
            return num_bytes, 'B '  # trailing ' ' for alignment
        elif num_bytes < 2**20:
            return num_bytes / 2**10, 'KB'
        elif num_bytes < 2**30:
            return num_bytes / 2**20, 'MB'
        else:
            return num_bytes / 2**30, 'GB'

    def update(self, advance_1: float | int, advance_2: float | int | None = None) -> None:
        self._create_task()
        if self.progress.finished:
            # nothing to report
            return

        self.total_1 += advance_1
        if self.unit_2 is not None:
            assert advance_2 is not None
            self.total_2 += advance_2

        total_1, unit_1 = (
            (self.total_1, self.unit_1) if self.unit_1 != 'B' else self._scale_bytes(cast(int, self.total_1))
        )

        total_2, unit_2 = (None, '') if self.unit_2 is None else (self.total_2, self.unit_2)
        if self.unit_2 == 'B':
            total_2, unit_2 = self._scale_bytes(cast(int, self.total_2))
        self.progress.update(
            self.task_id, total_1=_print_number(total_1), unit_1=unit_1, total_2=_print_number(total_2), unit_2=unit_2
        )

    def finalize(self) -> None:
        if self.progress.finished or self.task_id is None:
            # nothing to finalize
            return

        total_2_display = ''
        if self.unit_2 is not None:
            assert self.total_2 is not None
            total_2_display = _print_number(self.total_2)

        self.progress.update(self.task_id, total_1_display=_print_number(self.total_1), total_2_display=total_2_display)
