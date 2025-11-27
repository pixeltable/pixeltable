"""
OpenTelemetry provider setup for Pixeltable.

Manages TracerProvider and MeterProvider initialization with graceful handling
of existing providers (e.g., when Logfire or another SDK is already configured).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.trace import Tracer
    from opentelemetry.metrics import Meter

_logger = logging.getLogger('pixeltable')

DEFAULT_SERVICE_NAME = 'pixeltable'
ENV_ENABLED = 'PIXELTABLE_TELEMETRY_ENABLED'
ENV_SERVICE_NAME = 'OTEL_SERVICE_NAME'
ENV_OTLP_ENDPOINT = 'OTEL_EXPORTER_OTLP_ENDPOINT'


class TelemetryProvider:
    """
    Singleton provider for OpenTelemetry resources.

    Thread-safe initialization with support for external provider configuration
    (e.g., Logfire SDK configuring providers before Pixeltable).
    """

    _instance: ClassVar[TelemetryProvider | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _initialized: ClassVar[bool] = False

    __slots__ = (
        '_enabled',
        '_meter',
        '_meter_provider',
        '_otlp_endpoint',
        '_owns_meter_provider',
        '_owns_tracer_provider',
        '_service_name',
        '_tracer',
        '_tracer_provider',
    )

    def __init__(self) -> None:
        self._enabled: bool = False
        self._service_name: str = DEFAULT_SERVICE_NAME
        self._otlp_endpoint: str | None = None
        self._tracer_provider: TracerProvider | None = None
        self._meter_provider: MeterProvider | None = None
        self._tracer: Tracer | None = None
        self._meter: Meter | None = None
        self._owns_tracer_provider: bool = False
        self._owns_meter_provider: bool = False

    @classmethod
    def initialize(cls) -> TelemetryProvider:
        """Thread-safe singleton initialization."""
        if cls._initialized:
            return cls._instance  # type: ignore[return-value]

        with cls._lock:
            if cls._initialized:
                return cls._instance  # type: ignore[return-value]

            instance = cls()
            instance._load_config()
            if instance._enabled:
                instance._setup_providers()

            cls._instance = instance
            cls._initialized = True
            return instance

    @classmethod
    def get(cls) -> TelemetryProvider | None:
        """Get the singleton instance, initializing if needed."""
        if not cls._initialized:
            return cls.initialize()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the provider (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._shutdown()
            cls._instance = None
            cls._initialized = False

    def _load_config(self) -> None:
        """Load configuration from environment and config file."""
        # Environment variables (highest priority)
        env_enabled = os.environ.get(ENV_ENABLED, '').lower()
        env_endpoint = os.environ.get(ENV_OTLP_ENDPOINT)
        env_service_name = os.environ.get(ENV_SERVICE_NAME)

        # Config file (lower priority)
        config_enabled = None
        config_endpoint = None
        config_service_name = None

        try:
            from pixeltable.config import Config

            config = Config.get()
            config_enabled = config.get_bool_value('enabled', section='telemetry')
            config_endpoint = config.get_string_value('otlp_endpoint', section='telemetry')
            config_service_name = config.get_string_value('service_name', section='telemetry')
        except Exception:
            pass  # Config not available during early init

        # Determine enabled state
        if env_enabled in ('true', '1', 'yes'):
            self._enabled = True
        elif env_enabled in ('false', '0', 'no'):
            self._enabled = False
        elif config_enabled is not None:
            self._enabled = config_enabled
        else:
            # Auto-enable if endpoint is configured
            self._enabled = bool(env_endpoint or config_endpoint)

        self._otlp_endpoint = env_endpoint or config_endpoint
        self._service_name = env_service_name or config_service_name or DEFAULT_SERVICE_NAME

    def _setup_providers(self) -> None:
        """Set up OTEL providers, respecting existing configuration."""
        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            from opentelemetry.trace import NoOpTracerProvider
            from opentelemetry.metrics import NoOpMeterProvider

            resource = Resource.create({SERVICE_NAME: self._service_name})

            # Check if a real tracer provider is already configured
            current_tracer_provider = trace.get_tracer_provider()
            if isinstance(current_tracer_provider, NoOpTracerProvider):
                # No provider set yet - create our own
                self._tracer_provider = TracerProvider(resource=resource)
                exporter = OTLPSpanExporter(endpoint=self._otlp_endpoint) if self._otlp_endpoint else OTLPSpanExporter()
                self._tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                trace.set_tracer_provider(self._tracer_provider)
                self._owns_tracer_provider = True
            # else: External SDK (e.g., Logfire) already configured - use it

            self._tracer = trace.get_tracer('pixeltable')

            # Check if a real meter provider is already configured
            current_meter_provider = metrics.get_meter_provider()
            if isinstance(current_meter_provider, NoOpMeterProvider):
                if self._otlp_endpoint:
                    exporter = OTLPMetricExporter(endpoint=self._otlp_endpoint)
                else:
                    exporter = OTLPMetricExporter()
                reader = PeriodicExportingMetricReader(exporter)
                self._meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
                metrics.set_meter_provider(self._meter_provider)
                self._owns_meter_provider = True

            self._meter = metrics.get_meter('pixeltable')

        except ImportError as e:
            _logger.debug(f'OpenTelemetry setup skipped: {e}')
            self._enabled = False
        except Exception as e:
            _logger.warning(f'OpenTelemetry setup failed: {e}')
            self._enabled = False

    def _shutdown(self) -> None:
        """Shutdown providers we own."""
        if self._owns_tracer_provider and self._tracer_provider is not None:
            try:
                self._tracer_provider.shutdown()
            except Exception:
                pass

        if self._owns_meter_provider and self._meter_provider is not None:
            try:
                self._meter_provider.shutdown()
            except Exception:
                pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def tracer(self) -> Tracer | None:
        return self._tracer

    @property
    def meter(self) -> Meter | None:
        return self._meter

    @property
    def service_name(self) -> str:
        return self._service_name
