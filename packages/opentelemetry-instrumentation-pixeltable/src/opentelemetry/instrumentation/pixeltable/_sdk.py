"""Provider setup behind init().

init() runs once per process. Global providers are created at most once (OTEL providers are set-once per
process). Telemetry is exported to an OpenTelemetry Collector (or any OTLP endpoint) configured through the
standard `OTEL_EXPORTER_OTLP_*` environment variables; no vendor-specific behavior is baked in.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any, Literal

from opentelemetry import metrics as otel_metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.trace import ProxyTracerProvider

from pixeltable import exceptions as excs, telemetry
from pixeltable.config import Config
from pixeltable.env import Env

_logger = logging.getLogger('pixeltable.otel')

_SPAN_LEVELS = {'info': telemetry.INFO, 'debug': telemetry.DEBUG, 'trace': telemetry.TRACE}
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

    Call once, before the first Pixeltable operation. Each argument overrides the matching `[otel]`
    config setting and its standard `OTEL_*` environment variable; when an argument is left as None the
    value resolves from that env var (highest priority) then the `[otel]` config section. Pass
    `tracer_provider`/`meter_provider` to instrument against an SDK your application owns instead.

    Args:
        endpoint: OTLP collector endpoint (eg `http://localhost:4318`); resolves from
            `otel.exporter_otlp_endpoint` / `OTEL_EXPORTER_OTLP_ENDPOINT`. With no endpoint configured
            and no application-owned provider, instrumentation stays inert and exports nothing.
        protocol: OTLP transport, `http/protobuf` (default) or `grpc`; resolves from
            `otel.exporter_otlp_protocol` / `OTEL_EXPORTER_OTLP_PROTOCOL`.
        service_name: `service.name` resource attribute (default `pixeltable`); resolves from
            `otel.service_name` / `OTEL_SERVICE_NAME`.
        headers: OTLP headers as comma-separated `key=value` pairs; resolves from
            `otel.exporter_otlp_headers` / `OTEL_EXPORTER_OTLP_HEADERS`.
        span_level: span emission threshold: `info` (default; operation-level spans only), `debug`
            (adds per-row and per-UDF spans), or `trace`.
        metrics: force metric export on/off (by default metrics are exported only when an OTLP endpoint
            is configured).
        logs: force log export on/off (default off; must be explicitly enabled).
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
    Env.get().require_package('opentelemetry.instrumentation.fastapi')
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
    Env.get().require_package('opentelemetry.instrumentation.sqlalchemy')
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
    Env.get().require_package('opentelemetry.sdk')
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from . import PixeltableInstrumentor

    if span_level is None:
        span_level = config.get_string_value('span_level', section='otel') or 'info'
    # applied by instrument() below; validate now, before the set-once providers are mutated
    _resolve_span_level(span_level)

    cfg_endpoint = (
        endpoint if endpoint is not None else config.get_string_value('exporter_otlp_endpoint', section='otel')
    )
    cfg_protocol = (
        protocol if protocol is not None else config.get_string_value('exporter_otlp_protocol', section='otel')
    )
    cfg_service = (
        service_name
        if service_name is not None
        else (config.get_string_value('service_name', section='otel') or _DEFAULT_SERVICE_NAME)
    )
    cfg_headers = headers if headers is not None else config.get_string_value('exporter_otlp_headers', section='otel')
    if metrics is None:
        metrics = config.get_bool_value('metrics', section='otel')
    if logs is None:
        logs = config.get_bool_value('logs', section='otel')

    use_grpc = _use_grpc(cfg_protocol)

    # construct everything fallible before mutating any global state: providers are set-once and can't be
    # rolled back, so a partially-applied _setup() would leave init() unretryable and self-inconsistent
    owns_tp = False
    tp = tracer_provider
    if tp is None:
        existing = trace.get_tracer_provider()
        if not isinstance(existing, ProxyTracerProvider):
            # the embedding application already configured a tracer provider; never clobber it
            tp = existing
        elif cfg_endpoint is not None:
            exporter = _span_exporter(use_grpc, cfg_endpoint, cfg_headers)
            tp = TracerProvider(resource=_create_resource(cfg_service))
            tp.add_span_processor(BatchSpanProcessor(exporter))
            owns_tp = True
        # else: no endpoint configured and no app-owned provider -> stay inert; the bridge instruments
        # against the global no-op tracer and nothing is exported

    # Metrics flow when an OTLP endpoint is configured unless explicitly disabled.
    export_metrics = metrics is True or (metrics is not False and cfg_endpoint is not None)
    owns_mp = False
    mp = meter_provider
    if mp is None and export_metrics:
        existing_mp = otel_metrics.get_meter_provider()
        if 'Proxy' not in type(existing_mp).__name__:
            mp = existing_mp
        else:
            metric_exporter = _metric_exporter(use_grpc, cfg_endpoint, cfg_headers)
            mp = MeterProvider(
                resource=_create_resource(cfg_service), metric_readers=[PeriodicExportingMetricReader(metric_exporter)]
            )
            owns_mp = True

    # An endpoint alone must not export application logs; logs require explicit opt-in.
    export_logs = logs is True
    logger_provider: Any = None
    log_handler: logging.Handler | None = None
    if export_logs:
        logger_provider, log_handler = _build_log_export(use_grpc, cfg_endpoint, cfg_headers, cfg_service)

    # construction succeeded; apply global state
    if owns_tp:
        trace.set_tracer_provider(tp)
    if owns_mp:
        otel_metrics.set_meter_provider(mp)
    if logger_provider is not None:
        set_logger_provider(logger_provider)
        logging.getLogger('pixeltable').addHandler(log_handler)
    _state.owns_tracer_provider = owns_tp
    _state.owns_meter_provider = owns_mp
    _state.tracer_provider = tp
    _state.meter_provider = mp
    _state.logger_provider = logger_provider

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
    """Whether to use the OTLP/gRPC exporter rather than OTLP/HTTP; default http/protobuf."""
    protocol = (cfg_protocol or 'http/protobuf').strip().lower()
    return protocol == 'grpc'


def _span_exporter(use_grpc: bool, endpoint: str | None, headers: str | None) -> Any:
    Env.get().require_package(
        'opentelemetry.exporter.otlp.proto.grpc' if use_grpc else 'opentelemetry.exporter.otlp.proto.http'
    )
    hdrs = _parse_headers(headers) if headers is not None else None
    if use_grpc:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcSpanExporter

        return GrpcSpanExporter(endpoint=endpoint, headers=hdrs)
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpSpanExporter

    return HttpSpanExporter(
        endpoint=_join_endpoint(endpoint, 'v1/traces') if endpoint is not None else None, headers=hdrs
    )


def _metric_exporter(use_grpc: bool, endpoint: str | None, headers: str | None) -> Any:
    Env.get().require_package(
        'opentelemetry.exporter.otlp.proto.grpc' if use_grpc else 'opentelemetry.exporter.otlp.proto.http'
    )
    hdrs = _parse_headers(headers) if headers is not None else None
    if use_grpc:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as GrpcMetricExporter

        return GrpcMetricExporter(endpoint=endpoint, headers=hdrs)
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as HttpMetricExporter

    return HttpMetricExporter(
        endpoint=_join_endpoint(endpoint, 'v1/metrics') if endpoint is not None else None, headers=hdrs
    )


def _log_exporter(use_grpc: bool, endpoint: str | None, headers: str | None) -> Any:
    Env.get().require_package(
        'opentelemetry.exporter.otlp.proto.grpc' if use_grpc else 'opentelemetry.exporter.otlp.proto.http'
    )
    hdrs = _parse_headers(headers) if headers is not None else None
    if use_grpc:
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as GrpcLogExporter

        return GrpcLogExporter(endpoint=endpoint, headers=hdrs)
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as HttpLogExporter

    return HttpLogExporter(endpoint=_join_endpoint(endpoint, 'v1/logs') if endpoint is not None else None, headers=hdrs)


def _build_log_export(
    use_grpc: bool, cfg_endpoint: str | None, cfg_headers: str | None, cfg_service: str
) -> tuple[Any, logging.Handler]:
    Env.get().require_package('opentelemetry.sdk')
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    log_exporter = _log_exporter(use_grpc, cfg_endpoint, cfg_headers)
    logger_provider = LoggerProvider(resource=_create_resource(cfg_service))
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    return logger_provider, LoggingHandler(logger_provider=logger_provider)


def _create_resource(service_name: str) -> Any:
    Env.get().require_package('opentelemetry.sdk')
    from opentelemetry.sdk.resources import Resource

    return Resource.create({'service.name': service_name})


def _parse_headers(headers: str) -> dict[str, str]:
    """Parse comma-separated 'key=value' pairs (the OTEL_EXPORTER_OTLP_HEADERS format)."""
    from opentelemetry.util.re import parse_env_headers

    return dict(parse_env_headers(headers, liberal=True))


def _join_endpoint(endpoint: str, path: str) -> str:
    """Append a signal path to a base OTLP/HTTP endpoint unless it already ends with it."""
    base = endpoint.rstrip('/')
    if base.endswith(f'/{path}'):
        return endpoint
    return f'{base}/{path}'
