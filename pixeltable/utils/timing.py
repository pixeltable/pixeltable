"""Timing utilities for profiling insert path performance."""

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Generator

_timings: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)
_enabled: bool = False


def enable() -> None:
    """Enable timing instrumentation."""
    global _enabled  # noqa: PLW0603
    _enabled = True


def disable() -> None:
    """Disable timing instrumentation."""
    global _enabled  # noqa: PLW0603
    _enabled = False


def is_enabled() -> bool:
    """Check if timing is enabled."""
    return _enabled


@contextmanager
def timed(name: str) -> Generator[None, None, None]:
    """Context manager to measure execution time of a code block."""
    if not _enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        _timings[name] += time.perf_counter() - start
        _counts[name] += 1


def report() -> None:
    """Print timing report."""
    if not _timings:
        print('\n=== Insert Path Timing: No data collected ===')
        return

    print('\n=== Insert Path Timing ===')
    print(f'{"Name":<50} {"Total (s)":>12} {"Calls":>10} {"Avg (ms)":>12}')
    print('-' * 86)
    for name in sorted(_timings.keys()):
        total = _timings[name]
        count = _counts[name]
        avg_ms = (total / count * 1000) if count > 0 else 0
        print(f'{name:<50} {total:>12.3f} {count:>10} {avg_ms:>12.3f}')


def reset() -> None:
    """Reset all timing data."""
    _timings.clear()
    _counts.clear()
