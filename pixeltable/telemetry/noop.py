"""
No-op implementations for when OpenTelemetry is not available.

Provides the same interface but does nothing, enabling graceful degradation
without conditional checks throughout the codebase.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator


class NoOpSpanContext:
    """No-op span context that silently ignores all operations."""

    __slots__ = ()

    def __enter__(self) -> NoOpSpanContext:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status_ok(self) -> None:
        pass

    def set_status_error(self, description: str | None = None) -> None:
        pass


@contextmanager
def start_span(
    name: str,
    *,
    operation: str | None = None,
    table: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Generator[NoOpSpanContext, None, None]:
    """No-op span context manager."""
    yield NoOpSpanContext()
