"""
OpenTelemetry integration for Pixeltable.

This module provides observability through traces and metrics using OpenTelemetry.
Telemetry is opt-in and gracefully degrades when opentelemetry packages are not installed.

Configuration:
    Set environment variables or config.toml options:
    - OTEL_EXPORTER_OTLP_ENDPOINT or telemetry.otlp_endpoint
    - OTEL_SERVICE_NAME or telemetry.service_name (default: "pixeltable")
    - PIXELTABLE_TELEMETRY_ENABLED or telemetry.enabled (default: auto-detect from OTEL_* vars)
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tracing import SpanContext

# Thread-safe initialization
_init_lock = threading.Lock()
_initialized = False
_otel_available = False


def _check_otel_available() -> bool:
    """Check if OpenTelemetry SDK is importable."""
    try:
        import opentelemetry.sdk  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_initialized() -> None:
    """Thread-safe lazy initialization."""
    global _initialized, _otel_available
    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return
        _otel_available = _check_otel_available()
        if _otel_available:
            from .provider import TelemetryProvider
            TelemetryProvider.initialize()
        _initialized = True


def is_available() -> bool:
    """Check if OpenTelemetry packages are installed."""
    _ensure_initialized()
    return _otel_available


def is_enabled() -> bool:
    """Check if telemetry is enabled and configured."""
    _ensure_initialized()
    if not _otel_available:
        return False
    from .provider import TelemetryProvider
    provider = TelemetryProvider.get()
    return provider is not None and provider.enabled


def start_span(
    name: str,
    *,
    operation: str | None = None,
    table: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[SpanContext]:
    """
    Start a new span for tracing an operation.

    Args:
        name: Name of the span (e.g., "Query.collect", "Table.insert")
        operation: The database operation type (query, insert, update, delete)
        table: The table name being operated on
        attributes: Additional span attributes

    Yields:
        SpanContext for setting attributes and recording events
    """
    _ensure_initialized()
    if _otel_available:
        from .tracing import start_span as _start_span
        return _start_span(name, operation=operation, table=table, attributes=attributes)
    else:
        from .noop import start_span as _noop_start_span
        return _noop_start_span(name, operation=operation, table=table, attributes=attributes)


def record_query_duration(
    duration_seconds: float,
    *,
    table: str | None = None,
    query_type: str | None = None,
) -> None:
    """Record the duration of a query operation."""
    if not is_enabled():
        return
    from .metrics import record_query_duration as _record
    _record(duration_seconds, table=table, query_type=query_type)


def record_rows_processed(
    count: int,
    *,
    table: str | None = None,
    operation: str = 'query',
) -> None:
    """Record the number of rows processed by an operation."""
    if not is_enabled():
        return
    from .metrics import record_rows_processed as _record
    _record(count, table=table, operation=operation)


def record_udf_duration(
    duration_seconds: float,
    *,
    udf_name: str | None = None,
    batch_size: int | None = None,
) -> None:
    """Record the duration of a UDF execution."""
    if not is_enabled():
        return
    from .metrics import record_udf_duration as _record
    _record(duration_seconds, udf_name=udf_name, batch_size=batch_size)


def record_udf_error(
    *,
    udf_name: str | None = None,
    error_type: str | None = None,
) -> None:
    """Record a UDF error."""
    if not is_enabled():
        return
    from .metrics import record_udf_error as _record
    _record(udf_name=udf_name, error_type=error_type)


__all__ = [
    'is_available',
    'is_enabled',
    'start_span',
    'record_query_duration',
    'record_rows_processed',
    'record_udf_duration',
    'record_udf_error',
]
