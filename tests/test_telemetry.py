"""Tests for the telemetry module."""

from __future__ import annotations

import os
from unittest import mock

import pytest


def _otel_available() -> bool:
    """Check if OpenTelemetry SDK is available."""
    try:
        import opentelemetry.sdk  # noqa: F401
        return True
    except ImportError:
        return False


class TestTelemetryAvailability:
    """Tests for telemetry availability checks."""

    def test_is_available_returns_bool(self) -> None:
        """Test is_available returns bool based on OTEL installation."""
        from pixeltable import telemetry
        result = telemetry.is_available()
        assert isinstance(result, bool)

    def test_is_enabled_without_config(self) -> None:
        """Test is_enabled returns False when not configured."""
        from pixeltable import telemetry
        from pixeltable.telemetry import provider

        provider.TelemetryProvider.reset()

        # Clear OTEL env vars
        env_without_otel = {
            k: v for k, v in os.environ.items()
            if not k.startswith('OTEL_') and not k.startswith('PIXELTABLE_TELEMETRY')
        }
        with mock.patch.dict(os.environ, env_without_otel, clear=True):
            provider.TelemetryProvider.reset()
            result = telemetry.is_enabled()
            assert isinstance(result, bool)


class TestNoOpImplementations:
    """Tests for no-op implementations when OTEL is not available."""

    def test_noop_span_context(self) -> None:
        """Test NoOpSpanContext methods are no-ops."""
        from pixeltable.telemetry.noop import NoOpSpanContext

        ctx = NoOpSpanContext()

        # All methods should complete without error
        ctx.set_attribute('key', 'value')
        ctx.set_attributes({'key1': 'value1'})
        ctx.add_event('event_name', {'attr': 'value'})
        ctx.record_exception(ValueError('test'))
        ctx.set_status_ok()
        ctx.set_status_error('error')

        # Context manager should work
        with ctx:
            pass

    def test_noop_start_span(self) -> None:
        """Test no-op start_span context manager."""
        from pixeltable.telemetry.noop import start_span

        with start_span('test', operation='test', table='t') as span:
            span.set_attribute('key', 'value')


class TestTelemetryModuleInterface:
    """Tests for the telemetry module public interface."""

    def test_start_span_returns_context(self) -> None:
        """Test start_span returns a usable context."""
        from pixeltable import telemetry

        with telemetry.start_span('test_span') as span:
            assert span is not None
            span.set_attribute('test_key', 'test_value')

    def test_record_functions_callable(self) -> None:
        """Test metric recording functions are callable without error."""
        from pixeltable import telemetry

        # All should complete without error
        telemetry.record_query_duration(1.0, table='test', query_type='select')
        telemetry.record_rows_processed(100, table='test', operation='query')
        telemetry.record_udf_duration(0.5, udf_name='test_udf')
        telemetry.record_udf_error(udf_name='test_udf', error_type='ValueError')


@pytest.mark.skipif(not _otel_available(), reason='OpenTelemetry not installed')
class TestSpanContext:
    """Tests for SpanContext when OTEL is available."""

    def test_span_context_sets_attributes(self) -> None:
        """Test SpanContext attribute setting."""
        from pixeltable.telemetry.tracing import SpanContext
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        # Use a separate tracer for this test
        tracer = provider.get_tracer('test')
        with tracer.start_as_current_span('test_span') as span:
            ctx = SpanContext(span)
            ctx.set_attribute('test_key', 'test_value')
            ctx.set_attributes({'key1': 'value1', 'key2': 123})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get('test_key') == 'test_value'
        assert spans[0].attributes.get('key1') == 'value1'
        assert spans[0].attributes.get('key2') == 123


class TestConfiguration:
    """Tests for telemetry configuration."""

    def test_config_options_registered(self) -> None:
        """Test telemetry config options are registered."""
        from pixeltable.config import KNOWN_CONFIG_OPTIONS

        assert 'telemetry' in KNOWN_CONFIG_OPTIONS
        opts = KNOWN_CONFIG_OPTIONS['telemetry']
        assert 'enabled' in opts
        assert 'service_name' in opts
        assert 'otlp_endpoint' in opts

    def test_provider_respects_env_config(self) -> None:
        """Test provider loads configuration from environment."""
        from pixeltable.telemetry import provider

        provider.TelemetryProvider.reset()

        with mock.patch.dict(os.environ, {
            'OTEL_SERVICE_NAME': 'test-service',
            'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4317',
            'PIXELTABLE_TELEMETRY_ENABLED': 'true',
        }):
            provider.TelemetryProvider.reset()
            p = provider.TelemetryProvider.get()
            assert p is not None
            assert p.service_name == 'test-service'

        provider.TelemetryProvider.reset()
