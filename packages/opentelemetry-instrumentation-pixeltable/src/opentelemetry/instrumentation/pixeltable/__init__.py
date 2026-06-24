"""OpenTelemetry instrumentation for Pixeltable.

Translates Pixeltable's instrumentation hooks into OpenTelemetry spans and metrics. Instrumentation is
opt-in: either attach to an existing OpenTelemetry SDK with
[PixeltableInstrumentor][opentelemetry.instrumentation.pixeltable.PixeltableInstrumentor], or let
[init][opentelemetry.instrumentation.pixeltable.init] build providers from Pixeltable's `[otel]` config.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Collection
from typing import Any

from opentelemetry import context as otel_context, metrics as otel_metrics, trace

# opentelemetry-instrumentation marks instrumentor.py with a module-level `# type: ignore`, so mypy
# can't see BaseInstrumentor despite the package shipping py.typed
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore[attr-defined]
from opentelemetry.trace import StatusCode, set_span_in_context

from pixeltable import __version__, hooks

from ._sdk import init
from .package import _instruments

__all__ = ['PixeltableInstrumentor', 'init']

_ATTR_TYPES = (str, bool, int, float)


class _TraceContextFilter(logging.Filter):
    """Adds otelTraceID/otelSpanID attributes to records so logs are filterable by trace."""

    def filter(self, record: logging.LogRecord) -> bool:
        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            record.otelTraceID = trace.format_trace_id(span_context.trace_id)
            record.otelSpanID = trace.format_span_id(span_context.span_id)
        return True


_trace_context_filter = _TraceContextFilter()


def _clean_attrs(attrs: dict[str, Any] | None) -> dict[str, Any]:
    """Drop None values and internal ('_'-prefixed) keys; coerce non-primitive values to str."""
    if not attrs:
        return {}
    result: dict[str, Any] = {}
    for k, v in attrs.items():
        if v is None or k.startswith('_'):
            continue
        if isinstance(v, _ATTR_TYPES):
            result[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, _ATTR_TYPES) for x in v):
            result[k] = list(v)
        else:
            result[k] = str(v)
    return result


@dataclasses.dataclass(slots=True)
class _SpanToken:
    span: trace.Span
    ctx_token: object | None


class _OtelSubscriber(hooks.Subscriber):
    def __init__(self, tracer_provider: Any | None, meter_provider: Any | None) -> None:
        self._tracer = trace.get_tracer('pixeltable', __version__, tracer_provider=tracer_provider)
        meter = otel_metrics.get_meter('pixeltable', __version__, meter_provider=meter_provider)
        self._rows_written = meter.create_counter('pixeltable.rows.written', unit='{row}')
        self._cells_computed = meter.create_counter('pixeltable.cells.computed', unit='{cell}')
        self._cell_errors = meter.create_counter('pixeltable.cell.errors', unit='{error}')
        self._udf_calls = meter.create_counter('pixeltable.udf.calls', unit='{call}')
        self._udf_duration = meter.create_histogram('pixeltable.udf.duration', unit='s')
        self._xact_retries = meter.create_counter('pixeltable.xact.retries', unit='{retry}')

    def on_span_start(self, name: str, parent_token: Any, attrs: dict[str, Any] | None, set_current: bool) -> Any:
        if parent_token is not None:
            ctx = set_span_in_context(parent_token.span)
        else:
            ctx = otel_context.get_current()
        span = self._tracer.start_span(name, context=ctx, attributes=_clean_attrs(attrs))
        # set_current spans start and end in the same thread/context (hub contract), so attach/detach pair up
        ctx_token = otel_context.attach(set_span_in_context(span)) if set_current else None
        return _SpanToken(span, ctx_token)

    def on_span_end(self, token: Any, exc: BaseException | None, attrs: dict[str, Any] | None) -> None:
        span: trace.Span = token.span
        for k, v in _clean_attrs(attrs).items():
            span.set_attribute(k, v)
        if exc is not None:
            if isinstance(exc, Exception):
                span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
        span.end()
        if token.ctx_token is not None:
            otel_context.detach(token.ctx_token)

    def on_event(self, name: str, attrs: dict[str, Any] | None) -> None:
        attrs = attrs or {}
        if name == 'rows.written':
            self._rows_written.add(attrs.get('count', 1), _clean_attrs({'pxt.table': attrs.get('pxt.table')}))
        elif name == 'cells.computed':
            self._cells_computed.add(attrs.get('count', 1), _clean_attrs({'pxt.table': attrs.get('pxt.table')}))
        elif name == 'cell.error':
            self._cell_errors.add(
                attrs.get('count', 1),
                _clean_attrs({k: attrs.get(k) for k in ('pxt.table', 'pxt.column', 'error_type')}),
            )
        elif name == 'udf.stats':
            # the authoritative per-operation call count (udf.call events only cover scheduler calls)
            self._udf_calls.add(
                attrs.get('count', 0), _clean_attrs({k: attrs.get(k) for k in ('pxt.udf', 'pxt.column', 'pxt.table')})
            )
        elif name == 'udf.call':
            self._udf_duration.record(
                attrs.get('duration_s', 0.0),
                _clean_attrs({k: attrs.get(k) for k in ('pxt.udf', 'pxt.column', 'pxt.resource_pool', 'model')}),
            )
        elif name == 'xact.retry':
            self._xact_retries.add(1, _clean_attrs(attrs))

    def capture_context(self) -> Any:
        return otel_context.get_current()

    def restore_context(self, ctx: Any) -> Any:
        return otel_context.attach(ctx) if ctx is not None else None

    def exit_context(self, token: Any) -> None:
        if token is not None:
            otel_context.detach(token)


class PixeltableInstrumentor(BaseInstrumentor):
    """Enables OpenTelemetry instrumentation of pixeltable against the given (or global) providers.

    Example:

        >>> from opentelemetry.instrumentation.pixeltable import (
        ...     PixeltableInstrumentor,
        ... )
        ...
        ... PixeltableInstrumentor().instrument()
    """

    _subscriber: _OtelSubscriber | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        self._subscriber = _OtelSubscriber(kwargs.get('tracer_provider'), kwargs.get('meter_provider'))
        hooks.subscribe(self._subscriber)
        logging.getLogger('pixeltable').addFilter(_trace_context_filter)

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._subscriber is not None:
            hooks.unsubscribe(self._subscriber)
            self._subscriber = None
        logging.getLogger('pixeltable').removeFilter(_trace_context_filter)
