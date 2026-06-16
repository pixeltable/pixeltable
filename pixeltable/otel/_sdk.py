"""SDK provider setup for default-on instrumentation.

`setup()` runs once per process, either from Env initialization (auto, default-on when the
`pixeltable[otel]` extra is installed) or from `pixeltable.otel.init()` (manual). Global providers are
created at most once and survive Env re-initialization (OTEL providers are set-once per process); Env
teardown only flushes them and detaches the bridge.
"""

from __future__ import annotations

import base64
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
_PROVIDERS = frozenset({'auto', 'phoenix', 'langfuse', 'logfire', 'grafana', 'otlp'})
_LANGFUSE_DEFAULT_HOST = 'https://cloud.langfuse.com'
_LOGFIRE_DEFAULT_ENDPOINT = 'https://logfire-us.pydantic.dev'
_GRAFANA_DEFAULT_ENDPOINT = 'http://localhost:4318'


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
    provider: str | None = None,
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
        provider: telemetry backend: 'auto' (default; Phoenix if installed, else OTLP), 'phoenix',
            'langfuse', 'logfire', 'grafana', or 'otlp'. 'langfuse' reads LANGFUSE_PUBLIC_KEY /
            LANGFUSE_SECRET_KEY / LANGFUSE_HOST and 'logfire' reads LOGFIRE_TOKEN from the environment.
        endpoint: OTLP collector endpoint; the default depends on `provider`.
        service_name: `service.name` resource attribute and Phoenix project name.
        headers: OTLP headers as comma-separated 'key=value' pairs.
        metrics: force metric export on/off (by default metrics are exported only when the backend
            ingests them, eg grafana).
        logs: force log export on/off (same default as `metrics`).
        tracer_provider: an existing TracerProvider to instrument against.
        meter_provider: an existing MeterProvider to instrument against.

    Example:

        >>> import pixeltable.otel
        ...
        ... pixeltable.otel.init(provider='grafana')
    """
    if _state.initialized:
        raise excs.RequestError(excs.ErrorCode.INVALID_STATE, 'pixeltable.otel is already initialized for this session')
    setup(
        Config.get(),
        user_call=True,
        provider=provider,
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
    provider: str | None = None,
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

    provider_name = (
        provider if provider is not None else (config.get_string_value('provider', section='otel') or 'auto')
    ).lower()
    if provider_name not in _PROVIDERS:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f"Invalid value for 'otel.provider': {provider_name!r} (expected one of {', '.join(sorted(_PROVIDERS))})",
        )

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

    # the provider preset supplies backend-specific endpoint/auth defaults and whether the backend
    # ingests metrics/logs; the standard env vars above still take precedence over the preset
    force_phoenix, cfg_endpoint, cfg_headers, traces_only = _resolve_provider(
        provider_name, cfg_endpoint, cfg_headers, std_traces_env
    )
    try:
        phoenix_available = find_spec('phoenix.otel') is not None
    except ModuleNotFoundError:
        # find_spec on a dotted name imports the parent; an absent `phoenix` raises instead of None
        phoenix_available = False
    if force_phoenix and not phoenix_available:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            "otel.provider is 'phoenix' but arize-phoenix-otel is not installed (install pixeltable[otel])",
        )

    tp = tracer_provider
    if tp is not None:
        _state.owns_tracer_provider = False
    else:
        existing = trace.get_tracer_provider()
        if not isinstance(existing, ProxyTracerProvider):
            # the embedding application already configured a tracer provider; never clobber it
            tp = existing
            _state.owns_tracer_provider = False
        elif force_phoenix is not False and phoenix_available and not std_traces_env:
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

    # traces-only backends (Phoenix, Langfuse) reject metrics/logs: an exporter pointed at their
    # endpoint would error on every export interval, so metrics/logs only flow when the backend
    # ingests them (eg grafana/otlp) or on an explicit otel.metrics/otel.logs = true
    metrics_env = std_traces_env or 'OTEL_EXPORTER_OTLP_METRICS_ENDPOINT' in os.environ
    export_metrics = metrics is True or (
        metrics is not False and not traces_only and (metrics_env or cfg_endpoint is not None)
    )
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
    export_logs = logs is True or (logs is not False and not traces_only and (logs_env or cfg_endpoint is not None))
    if export_logs:
        _set_up_log_export(cfg_endpoint, cfg_headers, cfg_service, logs_env)

    PixeltableInstrumentor().instrument(tracer_provider=tp, meter_provider=mp)
    _state.initialized = True
    _state.user_initialized = user_call
    _logger.info('OpenTelemetry instrumentation enabled')


def _resolve_provider(
    provider: str, cfg_endpoint: str | None, cfg_headers: str | None, std_traces_env: bool
) -> tuple[bool | None, str | None, str | None, bool]:
    """Apply a provider preset, returning (force_phoenix, endpoint, headers, traces_only).

    force_phoenix is True to require the Phoenix register path, False to force plain OTLP, or None for
    auto (Phoenix if importable). The preset never overrides standard OTEL_* env vars: when those are
    set, setup() withholds the returned endpoint/headers so the exporter's native resolution wins.
    """
    if provider == 'phoenix':
        return True, cfg_endpoint, cfg_headers, True
    if provider == 'grafana':
        return False, cfg_endpoint or _GRAFANA_DEFAULT_ENDPOINT, cfg_headers, False
    if provider == 'otlp':
        return False, cfg_endpoint, cfg_headers, False
    if provider == 'langfuse':
        if std_traces_env:
            # operator pointed OTLP env vars elsewhere; honor them, just force OTLP + traces-only
            return False, cfg_endpoint, cfg_headers, True
        host = os.environ.get('LANGFUSE_HOST') or cfg_endpoint or _LANGFUSE_DEFAULT_HOST
        endpoint = host if host.rstrip('/').endswith('/api/public/otel') else f'{host.rstrip("/")}/api/public/otel'
        headers = cfg_headers
        public_key, secret_key = os.environ.get('LANGFUSE_PUBLIC_KEY'), os.environ.get('LANGFUSE_SECRET_KEY')
        if headers is None and public_key and secret_key:
            token = base64.b64encode(f'{public_key}:{secret_key}'.encode()).decode()
            headers = f'Authorization=Basic {token}'
        return False, endpoint, headers, True
    if provider == 'logfire':
        # Logfire ingests all three signals; region defaults to US (override via otel.endpoint for EU)
        if std_traces_env:
            return False, cfg_endpoint, cfg_headers, False
        endpoint = cfg_endpoint or _LOGFIRE_DEFAULT_ENDPOINT
        headers = cfg_headers
        token = os.environ.get('LOGFIRE_TOKEN')
        if headers is None and token:
            headers = f'Authorization={token}'  # Logfire takes the raw write token, no Bearer prefix
        return False, endpoint, headers, False
    return None, cfg_endpoint, cfg_headers, False  # auto


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
