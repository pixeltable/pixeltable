"""SDK provider setup for default-on instrumentation.

`setup()` runs once per process, either from Env initialization (auto, default-on when the
`pixeltable[otel]` extra is installed) or from `pixeltable.otel.init()` (manual). Global providers are
created at most once and survive Env re-initialization (OTEL providers are set-once per process); Env
teardown only flushes them and detaches the bridge.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from importlib.util import find_spec
from typing import Any

from opentelemetry import metrics as otel_metrics, trace
from opentelemetry.trace import ProxyTracerProvider

from pixeltable import exceptions as excs, hooks
from pixeltable.config import Config

from ._bridge import PixeltableInstrumentor

_logger = logging.getLogger('pixeltable.otel')

_SPAN_LEVELS = {'info': hooks.INFO, 'debug': hooks.DEBUG, 'trace': hooks.TRACE}
_DEFAULT_SERVICE_NAME = 'pixeltable'


@dataclasses.dataclass
class _State:
    initialized: bool = False
    user_initialized: bool = False
    owns_tracer_provider: bool = False
    owns_meter_provider: bool = False
    tracer_provider: Any = None
    meter_provider: Any = None
    log_handlers: list[tuple[logging.Logger, logging.Handler]] = dataclasses.field(default_factory=list)


_state = _State()


def init(
    *,
    endpoint: str | None = None,
    service_name: str | None = None,
    headers: str | None = None,
    metrics: bool | None = None,
    logs: bool | None = None,
    tracer_provider: Any | None = None,
    meter_provider: Any | None = None,
) -> None:
    """Manually configure pixeltable's OpenTelemetry instrumentation.

    Call before the first Pixeltable operation; the automatic default-on initialization then becomes a
    no-op. Arguments override the `otel` config section. Pass `tracer_provider`/`meter_provider` to
    instrument against an SDK your application owns.

    Args:
        endpoint: OTLP collector endpoint; defaults to the `otel.endpoint` config value or the local
            Phoenix collector.
        service_name: `service.name` resource attribute and Phoenix project name.
        headers: OTLP headers as comma-separated 'key=value' pairs.
        metrics: force metric export on/off (by default metrics are exported only when an explicit OTLP
            endpoint is configured).
        logs: force log export on/off (same default as `metrics`).
        tracer_provider: an existing TracerProvider to instrument against.
        meter_provider: an existing MeterProvider to instrument against.

    Example:

        >>> import pixeltable.otel
        ...
        ... pixeltable.otel.init(endpoint='http://localhost:4318')
    """
    if _state.initialized:
        raise excs.RequestError(excs.ErrorCode.INVALID_STATE, 'pixeltable.otel is already initialized for this session')
    setup(
        Config.get(),
        user_call=True,
        endpoint=endpoint,
        service_name=service_name,
        headers=headers,
        metrics=metrics,
        logs=logs,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )


def setup(
    config: Config,
    *,
    user_call: bool = False,
    endpoint: str | None = None,
    service_name: str | None = None,
    headers: str | None = None,
    metrics: bool | None = None,
    logs: bool | None = None,
    tracer_provider: Any | None = None,
    meter_provider: Any | None = None,
) -> None:
    """Set up providers (once per process) and attach the hook bridge.

    Must not call Env.get(): this runs inside Env setup, while the Env singleton is still initializing.
    """
    if _state.initialized:
        # Env re-init: re-attach the bridge to the standing providers
        PixeltableInstrumentor().instrument(
            tracer_provider=_state.tracer_provider, meter_provider=_state.meter_provider
        )
        return

    level_name = (config.get_string_value('span_level', section='otel') or 'info').lower()
    if level_name not in _SPAN_LEVELS:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f"Invalid value for 'otel.span_level': {level_name!r} (expected 'info', 'debug', or 'trace')",
        )
    hooks.set_span_level(_SPAN_LEVELS[level_name])

    cfg_endpoint = endpoint if endpoint is not None else config.get_string_value('endpoint', section='otel')
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

    # standard env vars beat pixeltable config: operators reconfigure deployed apps through them
    std_traces_env = 'OTEL_EXPORTER_OTLP_ENDPOINT' in os.environ or 'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT' in os.environ
    phoenix_env = 'PHOENIX_COLLECTOR_ENDPOINT' in os.environ

    tp = tracer_provider
    if tp is not None:
        _state.owns_tracer_provider = False
    else:
        existing = trace.get_tracer_provider()
        if not isinstance(existing, ProxyTracerProvider):
            # the embedding application already configured a tracer provider; never clobber it
            tp = existing
            _state.owns_tracer_provider = False
        elif find_spec('phoenix.otel') is not None and not std_traces_env:
            # default path: Phoenix register semantics (endpoint falls back to PHOENIX_COLLECTOR_ENDPOINT
            # or the local Phoenix collector); auto_instrument activates any installed OpenInference
            # provider instrumentors (openai, anthropic, ...)
            from phoenix.otel import register

            tp = register(
                endpoint=cfg_endpoint if not phoenix_env else None,
                project_name=cfg_service if 'PHOENIX_PROJECT_NAME' not in os.environ else None,
                headers=(
                    _parse_headers(cfg_headers)
                    if cfg_headers is not None and 'PHOENIX_CLIENT_HEADERS' not in os.environ
                    else None
                ),
                batch=True,
                set_global_tracer_provider=True,
                verbose=False,
                auto_instrument=True,
            )
            _state.owns_tracer_provider = True
        else:
            # plain SDK with an OTLP/HTTP exporter; config-derived kwargs are withheld whenever the
            # standard env vars are set, so the exporter's native env resolution wins
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter = OTLPSpanExporter(
                endpoint=(
                    _join_endpoint(cfg_endpoint, 'v1/traces')
                    if cfg_endpoint is not None and not std_traces_env
                    else None
                ),
                headers=_parse_headers(cfg_headers) if cfg_headers is not None and not std_traces_env else None,
            )
            tp = TracerProvider(resource=_create_resource(cfg_service))
            tp.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(tp)
            _state.owns_tracer_provider = True
    _state.tracer_provider = tp

    # Phoenix ingests traces only: a metrics exporter pointed at the default endpoint would error on
    # every export interval, so metrics/logs only flow when an explicit OTLP endpoint exists (or on
    # explicit otel.metrics/otel.logs = true)
    metrics_env = std_traces_env or 'OTEL_EXPORTER_OTLP_METRICS_ENDPOINT' in os.environ
    export_metrics = metrics is True or (metrics is not False and (metrics_env or cfg_endpoint is not None))
    mp = meter_provider
    if mp is not None:
        _state.owns_meter_provider = False
    elif export_metrics:
        existing_mp = otel_metrics.get_meter_provider()
        if 'Proxy' not in type(existing_mp).__name__:
            mp = existing_mp
            _state.owns_meter_provider = False
        else:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            metric_exporter = OTLPMetricExporter(
                endpoint=(
                    _join_endpoint(cfg_endpoint, 'v1/metrics') if cfg_endpoint is not None and not metrics_env else None
                ),
                headers=_parse_headers(cfg_headers) if cfg_headers is not None and not metrics_env else None,
            )
            mp = MeterProvider(
                resource=_create_resource(cfg_service), metric_readers=[PeriodicExportingMetricReader(metric_exporter)]
            )
            otel_metrics.set_meter_provider(mp)
            _state.owns_meter_provider = True
    _state.meter_provider = mp

    logs_env = std_traces_env or 'OTEL_EXPORTER_OTLP_LOGS_ENDPOINT' in os.environ
    export_logs = logs is True or (logs is not False and (logs_env or cfg_endpoint is not None))
    if export_logs:
        _set_up_log_export(cfg_endpoint, cfg_headers, cfg_service, logs_env)

    PixeltableInstrumentor().instrument(tracer_provider=tp, meter_provider=mp)
    _state.initialized = True
    _state.user_initialized = user_call
    _logger.info('OpenTelemetry instrumentation enabled')


def _set_up_log_export(cfg_endpoint: str | None, cfg_headers: str | None, cfg_service: str, logs_env: bool) -> None:
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    log_exporter = OTLPLogExporter(
        endpoint=_join_endpoint(cfg_endpoint, 'v1/logs') if cfg_endpoint is not None and not logs_env else None,
        headers=_parse_headers(cfg_headers) if cfg_headers is not None and not logs_env else None,
    )
    logger_provider = LoggerProvider(resource=_create_resource(cfg_service))
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    set_logger_provider(logger_provider)
    handler = LoggingHandler(logger_provider=logger_provider)
    pxt_logger = logging.getLogger('pixeltable')
    pxt_logger.addHandler(handler)
    _state.log_handlers.append((pxt_logger, handler))


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
    """Append a signal path to a base OTLP endpoint unless it already carries one."""
    if endpoint.rstrip('/').endswith(('/v1/traces', '/v1/metrics', '/v1/logs')):
        return endpoint
    return f'{endpoint.rstrip("/")}/{path}'


def managed_log_handlers() -> list[tuple[logging.Logger, logging.Handler]]:
    """Log handlers created by setup(), handed to Env for lifecycle management. Cleared on handoff."""
    handlers = list(_state.log_handlers)
    _state.log_handlers.clear()
    return handlers


def on_env_teardown() -> None:
    """Flush owned providers and detach the bridge; providers survive Env re-initialization."""
    PixeltableInstrumentor().uninstrument()
    for provider, owned in (
        (_state.tracer_provider, _state.owns_tracer_provider),
        (_state.meter_provider, _state.owns_meter_provider),
    ):
        if owned and provider is not None:
            try:
                provider.force_flush()
            except Exception as e:
                _logger.debug(f'error flushing OpenTelemetry provider: {e!r}')
