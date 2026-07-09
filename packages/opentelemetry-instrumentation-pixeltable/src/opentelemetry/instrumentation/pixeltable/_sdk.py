"""Provider setup behind init().

init() runs once per process. Global providers are created at most once (OTEL providers are set-once per
process). Telemetry is exported to an OpenTelemetry Collector (or any OTLP endpoint) configured through the
standard `OTEL_EXPORTER_OTLP_*` environment variables; no vendor-specific behavior is baked in.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
from typing import Any, Literal

from opentelemetry import metrics as otel_metrics, trace
from opentelemetry.trace import ProxyTracerProvider

from pixeltable import exceptions as excs, hooks
from pixeltable.config import Config
from pixeltable.env import Env

_logger = logging.getLogger('pixeltable.otel')

_SPAN_LEVELS = {'info': hooks.INFO, 'debug': hooks.DEBUG, 'trace': hooks.TRACE}
_DEFAULT_SERVICE_NAME = 'pixeltable'


@dataclasses.dataclass
class _State:
    initialized: bool = False
    owns_tracer_provider: bool = False
    owns_meter_provider: bool = False
    tracer_provider: Any = None
    meter_provider: Any = None
    logger_provider: Any = None


_state = _State()
_setup_lock = threading.Lock()


def init(
    *,
    endpoint: str | None = None,
    protocol: str | None = None,
    service_name: str | None = None,
    headers: str | None = None,
    span_level: Literal['info', 'debug', 'trace'] | None = None,
    metrics: bool | None = None,
    logs: bool | None = None,
    tracer_provider: Any = None,
    meter_provider: Any = None,
) -> None:
    """Configure pixeltable's OpenTelemetry instrumentation and start emitting telemetry.

    Call once, before the first Pixeltable operation. Arguments override the `otel` config section.
    Telemetry is exported to an OpenTelemetry Collector (or any OTLP endpoint) configured through the
    standard `OTEL_EXPORTER_OTLP_*` environment variables; pass `tracer_provider`/`meter_provider` to
    instrument against an SDK your application owns instead.

    Args:
        endpoint: OTLP collector endpoint (eg `http://localhost:4318`). `OTEL_EXPORTER_OTLP_ENDPOINT`
            takes precedence. With no endpoint configured and no application-owned provider,
            instrumentation stays inert and exports nothing.
        protocol: OTLP transport, `http/protobuf` (default) or `grpc`. `OTEL_EXPORTER_OTLP_PROTOCOL`
            takes precedence.
        service_name: `service.name` resource attribute (default `pixeltable`). `OTEL_SERVICE_NAME`
            takes precedence.
        headers: OTLP headers as comma-separated `key=value` pairs. `OTEL_EXPORTER_OTLP_HEADERS` takes
            precedence.
        span_level: span emission threshold: `info` (default; operation-level spans only), `debug`
            (adds per-row and per-UDF spans), or `trace`.
        metrics: force metric export on/off (by default metrics are exported only when an OTLP endpoint
            is configured).
        logs: force log export on/off (same default as `metrics`).
        tracer_provider: an existing TracerProvider to instrument against.
        meter_provider: an existing MeterProvider to instrument against.

    Example:

        >>> import opentelemetry.instrumentation.pixeltable as pxt_otel
        ...
        ... pxt_otel.init(endpoint='http://localhost:4318')
    """
    with _setup_lock:
        if _state.initialized:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_STATE, 'pixeltable OpenTelemetry instrumentation is already initialized'
            )
        _setup(
            Config.get(),
            endpoint=endpoint,
            protocol=protocol,
            service_name=service_name,
            headers=headers,
            span_level=span_level,
            metrics=metrics,
            logs=logs,
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )


def instrument_fastapi(app: Any, **kwargs: Any) -> None:
    """Instrument a FastAPI application so pixeltable spans nest under its request spans.

    Passthrough to the `opentelemetry-instrumentation-fastapi` package (which must be installed), wired
    to the tracer provider configured by [init][opentelemetry.instrumentation.pixeltable.init].

    Args:
        app: the FastAPI application to instrument.
        kwargs: forwarded to `FastAPIInstrumentor.instrument_app()`.

    Example:

        >>> import opentelemetry.instrumentation.pixeltable as pxt_otel
        ...
        ... pxt_otel.init(endpoint='http://localhost:4318')
        ... pxt_otel.instrument_fastapi(app)
    """
    Env.get().require_package('opentelemetry-instrumentation-fastapi')
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore[import-untyped]
    kwargs.setdefault('tracer_provider', _state.tracer_provider)
    FastAPIInstrumentor.instrument_app(app, **kwargs)


def _instrument_sqlalchemy(**kwargs: Any) -> None:
    """Instrument SQLAlchemy so its query spans are exported alongside pixeltable spans.

    Passthrough to the `opentelemetry-instrumentation-sqlalchemy` package (which must be installed),
    wired to the tracer provider configured by [init][opentelemetry.instrumentation.pixeltable.init].

    Args:
        kwargs: forwarded to `SQLAlchemyInstrumentor().instrument()` (eg `engine`).

    Example:

        >>> import opentelemetry.instrumentation.pixeltable as pxt_otel
        ...
        ... pxt_otel.init(endpoint='http://localhost:4318')
        ... pxt_otel._instrument_sqlalchemy(engine=engine)
    """
    Env.get().require_package('opentelemetry-instrumentation-sqlalchemy')
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor  # type: ignore[import-untyped]
    kwargs.setdefault('tracer_provider', _state.tracer_provider)
    SQLAlchemyInstrumentor().instrument(**kwargs)


def _setup(
    config: Config,
    *,
    endpoint: str | None,
    protocol: str | None,
    service_name: str | None,
    headers: str | None,
    span_level: str | None,
    metrics: bool | None,
    logs: bool | None,
    tracer_provider: Any,
    meter_provider: Any,
) -> None:
    """Provider setup proper; runs once, under _setup_lock held by init()."""
    if span_level is None:
        span_level = config.get_string_value('span_level', section='otel') or 'info'
    # applied by instrument() below; validate now, before the set-once providers are mutated
    _resolve_span_level(span_level)

    cfg_endpoint = endpoint if endpoint is not None else config.get_string_value('endpoint', section='otel')
    cfg_protocol = protocol if protocol is not None else config.get_string_value('protocol', section='otel')
    cfg_service = (
        service_name
        if service_name is not None
        else (config.get_string_value('service_name', section='otel') or _DEFAULT_SERVICE_NAME)
    )
    cfg_headers = headers if headers is not None else config.get_string_value('headers', section='otel')
    if metrics is None:
        metrics = config.get_bool_value('metrics', section='otel')
    if logs is None:
        logs = config.get_bool_value('logs', section='otel')

    use_grpc = _use_grpc(cfg_protocol)

    # standard env vars beat pixeltable config: operators reconfigure deployed apps through them
    std_traces_env = 'OTEL_EXPORTER_OTLP_ENDPOINT' in os.environ or 'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT' in os.environ

    # construct everything fallible before mutating any global state: providers are set-once and can't be
    # rolled back, so a partially-applied _setup() would leave init() unretryable and self-inconsistent
    owns_tp = False
    tp = tracer_provider
    if tp is None:
        existing = trace.get_tracer_provider()
        if not isinstance(existing, ProxyTracerProvider):
            # the embedding application already configured a tracer provider; never clobber it
            tp = existing
        elif std_traces_env or cfg_endpoint is not None:
            # plain SDK with an OTLP exporter; config-derived kwargs are withheld whenever the standard
            # env vars are set, so the exporter's native env resolution wins
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter = _span_exporter(use_grpc, cfg_endpoint, cfg_headers, withhold=std_traces_env)
            tp = TracerProvider(resource=_create_resource(cfg_service))
            tp.add_span_processor(BatchSpanProcessor(exporter))
            owns_tp = True
        # else: no endpoint configured and no app-owned provider -> stay inert; the bridge instruments
        # against the global no-op tracer and nothing is exported

    # metrics/logs flow only when an OTLP endpoint is configured (standard env var or otel.endpoint),
    # or on an explicit otel.metrics/otel.logs = true
    metrics_env = std_traces_env or 'OTEL_EXPORTER_OTLP_METRICS_ENDPOINT' in os.environ
    export_metrics = metrics is True or (metrics is not False and (metrics_env or cfg_endpoint is not None))
    owns_mp = False
    mp = meter_provider
    if mp is None and export_metrics:
        existing_mp = otel_metrics.get_meter_provider()
        if 'Proxy' not in type(existing_mp).__name__:
            mp = existing_mp
        else:
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            metric_exporter = _metric_exporter(use_grpc, cfg_endpoint, cfg_headers, withhold=metrics_env)
            mp = MeterProvider(
                resource=_create_resource(cfg_service), metric_readers=[PeriodicExportingMetricReader(metric_exporter)]
            )
            owns_mp = True

    logs_env = std_traces_env or 'OTEL_EXPORTER_OTLP_LOGS_ENDPOINT' in os.environ
    export_logs = logs is True or (logs is not False and (logs_env or cfg_endpoint is not None))
    logger_provider: Any = None
    log_handler: logging.Handler | None = None
    if export_logs:
        logger_provider, log_handler = _build_log_export(use_grpc, cfg_endpoint, cfg_headers, cfg_service, logs_env)

    # construction succeeded; apply global state
    if owns_tp:
        trace.set_tracer_provider(tp)
    if owns_mp:
        otel_metrics.set_meter_provider(mp)
    if logger_provider is not None:
        from opentelemetry._logs import set_logger_provider

        set_logger_provider(logger_provider)
        logging.getLogger('pixeltable').addHandler(log_handler)
    _state.owns_tracer_provider = owns_tp
    _state.owns_meter_provider = owns_mp
    _state.tracer_provider = tp
    _state.meter_provider = mp
    _state.logger_provider = logger_provider

    from . import PixeltableInstrumentor

    PixeltableInstrumentor().instrument(tracer_provider=tp, meter_provider=mp, span_level=span_level)
    _state.initialized = True
    _logger.info('OpenTelemetry instrumentation enabled')


def _resolve_span_level(span_level: str) -> int:
    level_name = span_level.lower()
    if level_name not in _SPAN_LEVELS:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f"Invalid value for 'span_level': {span_level!r} (expected 'info', 'debug', or 'trace')",
        )
    return _SPAN_LEVELS[level_name]


def _use_grpc(cfg_protocol: str | None) -> bool:
    """Whether to use the OTLP/gRPC exporter rather than OTLP/HTTP.

    The standard OTEL_EXPORTER_OTLP_PROTOCOL env var wins over the otel.protocol config value; the
    default is http/protobuf.
    """
    protocol = (os.environ.get('OTEL_EXPORTER_OTLP_PROTOCOL') or cfg_protocol or 'http/protobuf').strip().lower()
    return protocol == 'grpc'


def _span_exporter(use_grpc: bool, endpoint: str | None, headers: str | None, *, withhold: bool) -> Any:
    ep = None if withhold else endpoint
    hdrs = _parse_headers(headers) if headers is not None and not withhold else None
    if use_grpc:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcSpanExporter

        return GrpcSpanExporter(endpoint=ep, headers=hdrs)
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpSpanExporter

    return HttpSpanExporter(endpoint=_join_endpoint(ep, 'v1/traces') if ep is not None else None, headers=hdrs)


def _metric_exporter(use_grpc: bool, endpoint: str | None, headers: str | None, *, withhold: bool) -> Any:
    ep = None if withhold else endpoint
    hdrs = _parse_headers(headers) if headers is not None and not withhold else None
    if use_grpc:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as GrpcMetricExporter

        return GrpcMetricExporter(endpoint=ep, headers=hdrs)
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as HttpMetricExporter

    return HttpMetricExporter(endpoint=_join_endpoint(ep, 'v1/metrics') if ep is not None else None, headers=hdrs)


def _log_exporter(use_grpc: bool, endpoint: str | None, headers: str | None, *, withhold: bool) -> Any:
    ep = None if withhold else endpoint
    hdrs = _parse_headers(headers) if headers is not None and not withhold else None
    if use_grpc:
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as GrpcLogExporter

        return GrpcLogExporter(endpoint=ep, headers=hdrs)
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as HttpLogExporter

    return HttpLogExporter(endpoint=_join_endpoint(ep, 'v1/logs') if ep is not None else None, headers=hdrs)


def _build_log_export(
    use_grpc: bool, cfg_endpoint: str | None, cfg_headers: str | None, cfg_service: str, logs_env: bool
) -> tuple[Any, logging.Handler]:
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    log_exporter = _log_exporter(use_grpc, cfg_endpoint, cfg_headers, withhold=logs_env)
    logger_provider = LoggerProvider(resource=_create_resource(cfg_service))
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    return logger_provider, LoggingHandler(logger_provider=logger_provider)


def _create_resource(service_name: str) -> Any:
    from opentelemetry.sdk.resources import Resource

    # OTEL_SERVICE_NAME (read natively by Resource.create) wins over pixeltable config
    return Resource.create({} if 'OTEL_SERVICE_NAME' in os.environ else {'service.name': service_name})


def _parse_headers(headers: str) -> dict[str, str]:
    """Parse comma-separated 'key=value' pairs (the OTEL_EXPORTER_OTLP_HEADERS format)."""
    result: dict[str, str] = {}
    for pair in headers.split(','):
        key, sep, value = pair.partition('=')
        if sep:
            result[key.strip()] = value.strip()
    return result


def _join_endpoint(endpoint: str, path: str) -> str:
    """Append a signal path to a base OTLP/HTTP endpoint unless it already carries one."""
    if endpoint.rstrip('/').endswith(('/v1/traces', '/v1/metrics', '/v1/logs')):
        return endpoint
    return f'{endpoint.rstrip("/")}/{path}'
