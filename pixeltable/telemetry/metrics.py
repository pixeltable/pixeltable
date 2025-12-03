"""
Metrics utilities for Pixeltable.

Provides metric instruments for recording operation statistics.
"""

from __future__ import annotations

import threading
from typing import Any

from .provider import TelemetryProvider

# Thread-safe lazy initialization
_init_lock = threading.Lock()
_initialized = False

# Metric instruments
_query_duration_histogram = None
_query_rows_counter = None
_udf_duration_histogram = None
_udf_calls_counter = None
_udf_errors_counter = None
_insert_rows_counter = None
_update_rows_counter = None
_delete_rows_counter = None


def _ensure_instruments() -> bool:
    """Thread-safe initialization of metric instruments."""
    global _initialized  # noqa: PLW0603
    global _query_duration_histogram, _query_rows_counter  # noqa: PLW0603
    global _udf_duration_histogram, _udf_calls_counter, _udf_errors_counter  # noqa: PLW0603
    global _insert_rows_counter, _update_rows_counter, _delete_rows_counter  # noqa: PLW0603

    if _initialized:
        provider = TelemetryProvider.get()
        return provider is not None and provider.enabled

    with _init_lock:
        if _initialized:
            provider = TelemetryProvider.get()
            return provider is not None and provider.enabled

        provider = TelemetryProvider.get()
        if provider is None or provider.meter is None:
            _initialized = True
            return False

        meter = provider.meter

        _query_duration_histogram = meter.create_histogram(
            name='pixeltable.query.duration', description='Query operation duration', unit='s'
        )
        _query_rows_counter = meter.create_counter(
            name='pixeltable.query.rows', description='Rows returned by queries', unit='rows'
        )
        _udf_duration_histogram = meter.create_histogram(
            name='pixeltable.udf.duration', description='UDF execution duration', unit='s'
        )
        _udf_calls_counter = meter.create_counter(
            name='pixeltable.udf.calls', description='UDF invocations', unit='calls'
        )
        _udf_errors_counter = meter.create_counter(
            name='pixeltable.udf.errors', description='UDF errors', unit='errors'
        )
        _insert_rows_counter = meter.create_counter(
            name='pixeltable.insert.rows', description='Rows inserted', unit='rows'
        )
        _update_rows_counter = meter.create_counter(
            name='pixeltable.update.rows', description='Rows updated', unit='rows'
        )
        _delete_rows_counter = meter.create_counter(
            name='pixeltable.delete.rows', description='Rows deleted', unit='rows'
        )

        _initialized = True
        return True


def record_query_duration(duration_seconds: float, *, table: str | None = None, query_type: str | None = None) -> None:
    """Record query operation duration."""
    if not _ensure_instruments() or _query_duration_histogram is None:
        return

    attrs: dict[str, Any] = {}
    if table:
        attrs['db.name'] = table
    if query_type:
        attrs['pixeltable.query.type'] = query_type

    _query_duration_histogram.record(duration_seconds, attributes=attrs)


def record_rows_processed(count: int, *, table: str | None = None, operation: str = 'query') -> None:
    """Record rows processed by an operation."""
    if not _ensure_instruments():
        return

    attrs: dict[str, Any] = {'db.operation': operation}
    if table:
        attrs['db.name'] = table

    counter = {
        'query': _query_rows_counter,
        'insert': _insert_rows_counter,
        'update': _update_rows_counter,
        'delete': _delete_rows_counter,
    }.get(operation)

    if counter is not None:
        counter.add(count, attributes=attrs)


def record_udf_duration(duration_seconds: float, *, udf_name: str | None = None, batch_size: int | None = None) -> None:
    """Record UDF execution duration."""
    if not _ensure_instruments() or _udf_duration_histogram is None:
        return

    attrs: dict[str, Any] = {}
    if udf_name:
        attrs['pixeltable.udf.name'] = udf_name
    if batch_size is not None:
        attrs['pixeltable.udf.batch_size'] = batch_size

    _udf_duration_histogram.record(duration_seconds, attributes=attrs)

    if _udf_calls_counter is not None:
        _udf_calls_counter.add(1, attributes=attrs)


def record_udf_error(*, udf_name: str | None = None, error_type: str | None = None) -> None:
    """Record a UDF error."""
    if not _ensure_instruments() or _udf_errors_counter is None:
        return

    attrs: dict[str, Any] = {}
    if udf_name:
        attrs['pixeltable.udf.name'] = udf_name
    if error_type:
        attrs['error.type'] = error_type

    _udf_errors_counter.add(1, attributes=attrs)
