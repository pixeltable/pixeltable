"""
Tracing utilities for Pixeltable.

Provides span creation and context management for instrumenting operations.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Any, Generator, Self

from opentelemetry.trace import Status, StatusCode, Span

from .provider import TelemetryProvider

# Semantic conventions (aligned with OTEL database conventions)
ATTR_DB_SYSTEM = 'db.system'
ATTR_DB_OPERATION = 'db.operation'
ATTR_DB_NAME = 'db.name'


class SpanContext:
    """
    Wrapper around OpenTelemetry Span with a simplified interface.

    Provides methods for setting attributes, recording events, and managing status.
    """

    __slots__ = ('_span',)

    def __init__(self, span: Span) -> None:
        self._span = span

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None
    ) -> None:
        if exc_val is not None:
            self._span.record_exception(exc_val)
            self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute (None values are ignored)."""
        if value is not None:
            self._span.set_attribute(key, value)

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """Set multiple span attributes."""
        for key, value in attributes.items():
            if value is not None:
                self._span.set_attribute(key, value)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        self._span.add_event(name, attributes=attributes)

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span."""
        self._span.record_exception(exception)

    def set_status_ok(self) -> None:
        """Set span status to OK."""
        self._span.set_status(Status(StatusCode.OK))

    def set_status_error(self, description: str | None = None) -> None:
        """Set span status to ERROR."""
        self._span.set_status(Status(StatusCode.ERROR, description))


@contextmanager
def start_span(
    name: str, *, operation: str | None = None, table: str | None = None, attributes: dict[str, Any] | None = None
) -> Generator[SpanContext, None, None]:
    """
    Start a new span for tracing an operation.

    Args:
        name: Span name (e.g., "Query.collect", "Table.insert")
        operation: Database operation type (query, insert, update, delete)
        table: Table name being operated on
        attributes: Additional span attributes

    Yields:
        SpanContext for the active span

    Example:
        with start_span("Query.collect", operation="query", table="my_table") as span:
            span.set_attribute("pixeltable.rows_processed", 100)
            # ... perform operation ...
    """
    provider = TelemetryProvider.get()
    if provider is None or provider.tracer is None:
        from .noop import NoOpSpanContext

        yield NoOpSpanContext()
        return

    # Build span attributes
    span_attrs: dict[str, Any] = {ATTR_DB_SYSTEM: 'pixeltable'}
    if operation:
        span_attrs[ATTR_DB_OPERATION] = operation
    if table:
        span_attrs[ATTR_DB_NAME] = table
    if attributes:
        span_attrs.update(attributes)

    with provider.tracer.start_as_current_span(name, attributes=span_attrs) as span:
        ctx = SpanContext(span)
        try:
            yield ctx
            ctx.set_status_ok()
        except Exception as e:
            ctx.record_exception(e)
            ctx.set_status_error(str(e))
            raise
