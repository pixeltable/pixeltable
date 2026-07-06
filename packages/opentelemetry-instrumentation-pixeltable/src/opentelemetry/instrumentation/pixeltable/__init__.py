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
from typing import Any, Callable

from opentelemetry import context as otel_context, metrics as otel_metrics, trace
from opentelemetry.context import Context, Token

# opentelemetry-instrumentation marks instrumentor.py with a module-level `# type: ignore`, so mypy
# can't see BaseInstrumentor despite the package shipping py.typed
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore[attr-defined]
from opentelemetry.trace import StatusCode, set_span_in_context

from pixeltable import __version__, hooks

from ._sdk import _resolve_span_level, init
from .package import _instruments

__all__ = ['PixeltableInstrumentor', 'init']

_ATTR_TYPES = (str, bool, int, float)


_prev_record_factory: Callable[..., logging.LogRecord] | None = None


def _install_record_factory() -> None:
    """Stamp pixeltable log records with otelTraceID/otelSpanID so logs are filterable by trace.

    A record factory (the mechanism opentelemetry-instrumentation-logging uses) rather than a filter on
    the 'pixeltable' logger: logging runs logger-level filters only on the originating logger, so a filter
    there never sees records created on child loggers (`pixeltable.exec...` etc.), which is where nearly
    all pixeltable records originate.
    """
    global _prev_record_factory  # noqa: PLW0603
    prev = logging.getLogRecordFactory()
    _prev_record_factory = prev

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = prev(*args, **kwargs)
        if record.name == 'pixeltable' or record.name.startswith('pixeltable.'):
            span_context = trace.get_current_span().get_span_context()
            if span_context.is_valid:
                record.otelTraceID = trace.format_trace_id(span_context.trace_id)
                record.otelSpanID = trace.format_span_id(span_context.span_id)
        return record

    logging.setLogRecordFactory(factory)


def _uninstall_record_factory() -> None:
    global _prev_record_factory  # noqa: PLW0603
    if _prev_record_factory is not None:
        logging.setLogRecordFactory(_prev_record_factory)
        _prev_record_factory = None


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
    ctx_token: Token[Context] | None


class _OtelSubscriber(hooks.Subscriber):
    def __init__(self, tracer_provider: Any | None, meter_provider: Any | None) -> None:
        self._tracer = trace.get_tracer('pixeltable', __version__, tracer_provider=tracer_provider)
        self._meter = otel_metrics.get_meter('pixeltable', __version__, meter_provider=meter_provider)
        # TODO(part 2): create the metric instruments here and translate events in on_event() once core
        # emits the corresponding hooks.emit() events

    def on_span_start(self, name: str, parent_token: Any, attrs: dict[str, Any] | None, set_current: bool) -> Any:
        if parent_token is not None:
            assert isinstance(parent_token, _SpanToken)
            ctx = set_span_in_context(parent_token.span)
        else:
            ctx = otel_context.get_current()
        span = self._tracer.start_span(name, context=ctx, attributes=_clean_attrs(attrs))
        # set_current spans start and end in the same thread/context (hub contract), so attach/detach pair up
        ctx_token = otel_context.attach(set_span_in_context(span)) if set_current else None
        return _SpanToken(span, ctx_token)

    def on_span_end(self, token: Any, exc: BaseException | None, attrs: dict[str, Any] | None) -> None:
        assert isinstance(token, _SpanToken)
        span = token.span
        for k, v in _clean_attrs(attrs).items():
            span.set_attribute(k, v)
        if exc is not None:
            if isinstance(exc, Exception):
                span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
        span.end()
        if token.ctx_token is not None:
            otel_context.detach(token.ctx_token)

    def capture_context(self) -> Any:
        return otel_context.get_current()

    def restore_context(self, ctx: Any) -> Any:
        return otel_context.attach(ctx) if ctx is not None else None

    def exit_context(self, token: Any) -> None:
        if token is not None:
            otel_context.detach(token)


class PixeltableInstrumentor(BaseInstrumentor):
    """Enables OpenTelemetry instrumentation of pixeltable against the given (or global) providers.

    `instrument()` accepts `tracer_provider`, `meter_provider`, and `span_level` (span emission
    threshold: `info` (default), `debug`, or `trace`).

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
        span_level = kwargs.get('span_level')
        if span_level is not None:
            hooks.set_span_level(_resolve_span_level(span_level))
        self._subscriber = _OtelSubscriber(kwargs.get('tracer_provider'), kwargs.get('meter_provider'))
        hooks.subscribe(self._subscriber)
        _install_record_factory()

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._subscriber is not None:
            hooks.unsubscribe(self._subscriber)
            self._subscriber = None
        _uninstall_record_factory()
