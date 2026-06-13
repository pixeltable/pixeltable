"""The hooks -> OpenTelemetry bridge: translates pixeltable instrumentation hooks into spans and metrics."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from opentelemetry import context as otel_context, metrics as otel_metrics, trace
from opentelemetry.trace import StatusCode, set_span_in_context

from pixeltable import __version__, hooks

from .logs import TRACE_CONTEXT_FILTER
from .usage import extract_usage

_ATTR_TYPES = (str, bool, int, float)


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
        self._udf_tokens = meter.create_counter('pixeltable.udf.tokens', unit='{token}')
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
            metric_attrs = _clean_attrs(
                {k: attrs.get(k) for k in ('pxt.udf', 'pxt.column', 'pxt.resource_pool', 'model')}
            )
            self._udf_duration.record(attrs.get('duration_s', 0.0), metric_attrs)
            model, tokens = extract_usage(attrs.get('_result'))
            if model is not None and 'model' not in metric_attrs:
                metric_attrs['model'] = model
            for token_type, count in tokens.items():
                self._udf_tokens.add(count, {**metric_attrs, 'type': token_type})
        elif name == 'xact.retry':
            self._xact_retries.add(1, _clean_attrs(attrs))

    def capture_context(self) -> Any:
        return otel_context.get_current()

    def restore_context(self, ctx: Any) -> Any:
        return otel_context.attach(ctx) if ctx is not None else None

    def exit_context(self, token: Any) -> None:
        if token is not None:
            otel_context.detach(token)


_active_subscriber: _OtelSubscriber | None = None


class PixeltableInstrumentor:
    """Enables OpenTelemetry instrumentation of pixeltable against the given (or global) providers.

    Mirrors the `opentelemetry-instrumentation` BaseInstrumentor interface without depending on it.

    Example:

        >>> from pixeltable.otel import PixeltableInstrumentor
        ...
        ... PixeltableInstrumentor().instrument()
    """

    def instrument(self, *, tracer_provider: Any | None = None, meter_provider: Any | None = None) -> None:
        global _active_subscriber  # noqa: PLW0603
        if _active_subscriber is not None:
            return
        _active_subscriber = _OtelSubscriber(tracer_provider, meter_provider)
        hooks.subscribe(_active_subscriber)
        logging.getLogger('pixeltable').addFilter(TRACE_CONTEXT_FILTER)

    def uninstrument(self) -> None:
        global _active_subscriber  # noqa: PLW0603
        if _active_subscriber is None:
            return
        hooks.unsubscribe(_active_subscriber)
        logging.getLogger('pixeltable').removeFilter(TRACE_CONTEXT_FILTER)
        _active_subscriber = None

    @property
    def is_instrumented(self) -> bool:
        return _active_subscriber is not None
