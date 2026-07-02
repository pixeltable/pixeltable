import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

import pixeltable as pxt


def _run_py(code: str, env_overrides: dict[str, str], tmp_home: Path) -> 'subprocess.CompletedProcess[str]':
    # isolate config/home: init() reads pixeltable config, and global OTEL providers are set-once per process
    env = {**os.environ, 'PIXELTABLE_HOME': str(tmp_home), **env_overrides}
    result = subprocess.run((sys.executable, '-c', code), capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, f'stderr:\n{result.stderr}\nstdout:\n{result.stdout}'
    return result


def test_no_otel_imports_in_core() -> None:
    # core pixeltable must never import opentelemetry; instrumentation lives only in this package
    pkg_root = Path(pxt.__file__).parent
    offenders = [
        str(path)
        for path in pkg_root.rglob('*.py')
        if re.search(r'^\s*(import opentelemetry|from opentelemetry)', path.read_text(), flags=re.MULTILINE)
    ]
    assert offenders == []


def test_import_does_not_load_opentelemetry(tmp_path: Path) -> None:
    # importing pixeltable never pulls in opentelemetry: instrumentation is strictly opt-in
    _run_py(
        'import pixeltable, sys\n'
        "assert not any(m == 'opentelemetry' or m.startswith('opentelemetry.') for m in sys.modules), "
        "[m for m in sys.modules if m.startswith('opentelemetry')]\n",
        {},
        tmp_path,
    )


class TestInit:
    """Subprocess-based tests: global OTEL providers are set-once per process."""

    def test_builds_tracer_provider(self, tmp_path: Path) -> None:
        _run_py(
            'import opentelemetry.instrumentation.pixeltable as o\n'
            'o.init()\n'
            'from opentelemetry import trace\n'
            'from opentelemetry.sdk.trace import TracerProvider\n'
            'tp = trace.get_tracer_provider()\n'
            'assert isinstance(tp, TracerProvider), type(tp)\n'
            "assert tp.resource.attributes['service.name'] == 'pixeltable'\n"
            'import opentelemetry.instrumentation.pixeltable._sdk as sdk\n'
            'assert sdk._state.initialized and sdk._state.owns_tracer_provider and sdk._state.owns_meter_provider\n'
            'assert o.PixeltableInstrumentor().is_instrumented_by_opentelemetry\n',
            {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://127.0.0.1:9', 'OTEL_EXPORTER_OTLP_TIMEOUT': '1'},
            tmp_path,
        )

    def test_inert_without_endpoint(self, tmp_path: Path) -> None:
        # no endpoint configured and no app-owned provider: the bridge attaches but nothing is exported
        _run_py(
            'import os\n'
            "for _k in [k for k in os.environ if k.startswith('OTEL_')]:\n"
            '    del os.environ[_k]\n'
            'import opentelemetry.instrumentation.pixeltable as o\n'
            'o.init()\n'
            'from opentelemetry import trace\n'
            'from opentelemetry.trace import ProxyTracerProvider\n'
            'assert isinstance(trace.get_tracer_provider(), ProxyTracerProvider)\n'
            'import opentelemetry.instrumentation.pixeltable._sdk as sdk\n'
            'assert sdk._state.initialized\n'
            'assert not sdk._state.owns_tracer_provider and sdk._state.tracer_provider is None\n'
            'assert o.PixeltableInstrumentor().is_instrumented_by_opentelemetry\n',
            {},
            tmp_path,
        )

    def test_respects_existing_sdk(self, tmp_path: Path) -> None:
        _run_py(
            'from opentelemetry import trace\n'
            'from opentelemetry.sdk.trace import TracerProvider\n'
            'my_tp = TracerProvider()\n'
            'trace.set_tracer_provider(my_tp)\n'
            'import opentelemetry.instrumentation.pixeltable as o\n'
            'o.init()\n'
            'assert trace.get_tracer_provider() is my_tp\n'
            'import opentelemetry.instrumentation.pixeltable._sdk as sdk\n'
            'assert sdk._state.initialized and not sdk._state.owns_tracer_provider\n',
            {},
            tmp_path,
        )

    def test_double_init_raises(self, tmp_path: Path) -> None:
        _run_py(
            'import pixeltable as pxt\n'
            'import opentelemetry.instrumentation.pixeltable as o\n'
            'o.init()\n'
            'try:\n'
            '    o.init()\n'
            "    raise AssertionError('expected init() to raise')\n"
            'except pxt.exceptions.RequestError:\n'
            '    pass\n',
            {'OTEL_EXPORTER_OTLP_TIMEOUT': '1'},
            tmp_path,
        )

    def test_protocol_grpc(self, tmp_path: Path) -> None:
        _run_py(
            'import opentelemetry.instrumentation.pixeltable as o\n'
            'o.init()\n'
            'from opentelemetry import trace\n'
            'tp = trace.get_tracer_provider()\n'
            'exporter = tp._active_span_processor._span_processors[0].span_exporter\n'
            "assert type(exporter).__module__.startswith('opentelemetry.exporter.otlp.proto.grpc'), "
            'type(exporter).__module__\n',
            {
                'OTEL_EXPORTER_OTLP_PROTOCOL': 'grpc',
                'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://127.0.0.1:4317',
                'OTEL_EXPORTER_OTLP_TIMEOUT': '1',
            },
            tmp_path,
        )


class TestProtocolSelection:
    """Unit tests for the OTLP transport selection (pure function, no providers touched)."""

    def test_default_is_http(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry.instrumentation.pixeltable._sdk import _use_grpc

        monkeypatch.delenv('OTEL_EXPORTER_OTLP_PROTOCOL', raising=False)
        assert _use_grpc(None) is False
        assert _use_grpc('http/protobuf') is False

    def test_grpc_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry.instrumentation.pixeltable._sdk import _use_grpc

        monkeypatch.delenv('OTEL_EXPORTER_OTLP_PROTOCOL', raising=False)
        assert _use_grpc('grpc') is True

    def test_env_overrides_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry.instrumentation.pixeltable._sdk import _use_grpc

        monkeypatch.setenv('OTEL_EXPORTER_OTLP_PROTOCOL', 'grpc')
        assert _use_grpc('http/protobuf') is True


class TestBridge:
    """In-process bridge correctness with in-memory exporters (no global providers touched)."""

    @pytest.fixture
    def instrumented(self) -> Iterator[tuple[Any, Any]]:
        from opentelemetry.instrumentation.pixeltable import PixeltableInstrumentor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])
        instrumentor = PixeltableInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)
        yield span_exporter, metric_reader
        instrumentor.uninstrument()

    @staticmethod
    def _metric_points(metric_reader: Any, name: str) -> list[Any]:
        data = metric_reader.get_metrics_data()
        points = []
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name == name:
                        points.extend(metric.data.data_points)
        return points

    def test_span_capture(self, instrumented: tuple[Any, Any]) -> None:
        from pixeltable import hooks

        span_exporter, _ = instrumented
        hooks.span_end(hooks.span_start('pixeltable.insert', set_current=True))
        assert [s.name for s in span_exporter.get_finished_spans()] == ['pixeltable.insert']

    def test_span_nesting(self, instrumented: tuple[Any, Any]) -> None:
        from pixeltable import hooks

        span_exporter, _ = instrumented
        parent = hooks.span_start('pixeltable.op', set_current=True)
        child = hooks.span_start('udf.foo', set_current=True)
        hooks.span_end(child)
        hooks.span_end(parent)
        spans = {s.name: s for s in span_exporter.get_finished_spans()}
        assert spans['udf.foo'].parent is not None
        assert spans['udf.foo'].parent.span_id == spans['pixeltable.op'].context.span_id

    def test_rows_written_metric(self, instrumented: tuple[Any, Any]) -> None:
        from pixeltable import hooks

        _, metric_reader = instrumented
        hooks.emit('rows.written', attrs={'count': 5, 'pxt.table': 'mytbl'})
        points = self._metric_points(metric_reader, 'pixeltable.rows.written')
        assert sum(p.value for p in points) == 5
        assert all(p.attributes['pxt.table'] == 'mytbl' for p in points)

    def test_udf_duration_metric(self, instrumented: tuple[Any, Any]) -> None:
        from pixeltable import hooks

        _, metric_reader = instrumented
        hooks.emit('udf.call', attrs={'pxt.udf': 'openai.chat_completions', 'duration_s': 0.5})
        points = self._metric_points(metric_reader, 'pixeltable.udf.duration')
        assert len(points) == 1
        assert points[0].sum == 0.5
        assert points[0].attributes['pxt.udf'] == 'openai.chat_completions'

    def test_uninstrument_stops_emission(self) -> None:
        from opentelemetry.instrumentation.pixeltable import PixeltableInstrumentor
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from pixeltable import hooks

        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        instrumentor = PixeltableInstrumentor()
        prev_factory = logging.getLogRecordFactory()
        instrumentor.instrument(tracer_provider=tracer_provider)
        try:
            hooks.span_end(hooks.span_start('pixeltable.op', set_current=True))
            assert len(span_exporter.get_finished_spans()) == 1
        finally:
            instrumentor.uninstrument()
        assert not hooks.active()
        assert logging.getLogRecordFactory() is prev_factory
        hooks.span_end(hooks.span_start('pixeltable.op', set_current=True))
        assert len(span_exporter.get_finished_spans()) == 1  # nothing new after uninstrument

    def test_log_correlation(self, instrumented: tuple[Any, Any]) -> None:
        from opentelemetry import trace

        from pixeltable import hooks

        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        pxt_logger = logging.getLogger('pixeltable')
        # log on a child logger: that's where real pixeltable records originate, and the record factory
        # (unlike a logger-level filter) must stamp those too
        child_logger = logging.getLogger('pixeltable.exec.test_child')
        handler = _Capture()
        pxt_logger.addHandler(handler)
        old_level = pxt_logger.level
        pxt_logger.setLevel(logging.INFO)
        try:
            span_handle = hooks.span_start('pixeltable.insert', set_current=True)
            try:
                child_logger.info('correlated message')
                ctx = trace.get_current_span().get_span_context()
                assert ctx.is_valid
            finally:
                hooks.span_end(span_handle)
            record = next(r for r in records if r.getMessage() == 'correlated message')
            assert record.otelTraceID == trace.format_trace_id(ctx.trace_id)  # type: ignore[attr-defined]
            assert record.otelSpanID == trace.format_span_id(ctx.span_id)  # type: ignore[attr-defined]
        finally:
            pxt_logger.removeHandler(handler)
            pxt_logger.setLevel(old_level)
