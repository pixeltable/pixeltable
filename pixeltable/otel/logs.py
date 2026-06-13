"""Log correlation: stamp the active trace/span ids onto pixeltable log records."""

from __future__ import annotations

import logging

from opentelemetry import trace


class TraceContextFilter(logging.Filter):
    """Adds otelTraceID/otelSpanID attributes to records so logs are filterable by trace."""

    def filter(self, record: logging.LogRecord) -> bool:
        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            record.otelTraceID = trace.format_trace_id(span_context.trace_id)
            record.otelSpanID = trace.format_span_id(span_context.span_id)
        return True


TRACE_CONTEXT_FILTER = TraceContextFilter()
