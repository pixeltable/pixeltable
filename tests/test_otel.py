import logging
import os
import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Iterator

import pytest

import pixeltable as pxt

_OTEL_INSTALLED = find_spec('opentelemetry.sdk') is not None


def _run_py(code: str, env_overrides: dict[str, str]) -> 'subprocess.CompletedProcess[str]':
    env = {**os.environ, **env_overrides}
    # a dedicated db per worker: Env teardown in a subprocess (eg, test_env_reinit) terminates all
    # connections to its db, which would kill the parent test session's connections
    env['PIXELTABLE_DB'] = f'{os.environ.get("PIXELTABLE_DB", "test")}_otel'
    result = subprocess.run((sys.executable, '-c', code), capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, f'stderr:\n{result.stderr}\nstdout:\n{result.stdout}'
    return result


class TestOtelInit:
    """Subprocess-based tests: global OTEL providers are set-once per process."""

    def test_import_isolation(self) -> None:
        # the headline guarantee: importing pixeltable never imports opentelemetry
        _run_py(
            'import pixeltable, sys\n'
            "assert not any(m == 'opentelemetry' or m.startswith('opentelemetry.') for m in sys.modules), "
            "[m for m in sys.modules if m.startswith('opentelemetry')]\n",
            {'PIXELTABLE_OTEL': '1'},
        )

    def test_no_otel_imports_in_core(self) -> None:
        # core pixeltable (everything outside pixeltable/otel/) must not import opentelemetry
        pkg_root = Path(pxt.__file__).parent
        offenders = []
        for path in pkg_root.rglob('*.py'):
            if pkg_root / 'otel' in path.parents:
                continue
            if re.search(r'^\s*(import opentelemetry|from opentelemetry)', path.read_text(), flags=re.MULTILINE):
                offenders.append(str(path))
        assert offenders == []

    @pytest.mark.skipif(not _OTEL_INSTALLED, reason='otel extra not installed')
    def test_default_on(self, init_env: None) -> None:
        _run_py(
            'import pixeltable as pxt\n'
            'pxt.init()\n'
            'from opentelemetry import trace\n'
            'from opentelemetry.sdk.trace import TracerProvider\n'
            'tp = trace.get_tracer_provider()\n'
            'assert isinstance(tp, TracerProvider), type(tp)\n'
            "assert tp.resource.attributes['service.name'] == 'pixeltable'\n"
            'import pixeltable.otel._sdk as sdk\n'
            'assert sdk._state.initialized and sdk._state.owns_tracer_provider\n'
            # an endpoint is configured, so metrics export and the meter provider is owned too
            'assert sdk._state.owns_meter_provider\n'
            'from pixeltable.otel import PixeltableInstrumentor\n'
            'assert PixeltableInstrumentor().is_instrumented\n',
            {
                'PIXELTABLE_OTEL': '1',
                # forces the plain-SDK path; nothing is listening, but no spans are produced either
                'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://127.0.0.1:9',
                'OTEL_EXPORTER_OTLP_TIMEOUT': '1',
            },
        )

    @pytest.mark.skipif(not _OTEL_INSTALLED, reason='otel extra not installed')
    def test_inert_without_endpoint(self, init_env: None) -> None:
        # default-on with no endpoint configured: the bridge attaches but nothing is exported
        _run_py(
            'import os\n'
            "for _k in [k for k in os.environ if k.startswith('OTEL_EXPORTER_OTLP') or k == 'OTEL_ENDPOINT']:\n"
            '    del os.environ[_k]\n'
            'import pixeltable as pxt\n'
            'pxt.init()\n'
            'from opentelemetry import trace\n'
            'from opentelemetry.trace import ProxyTracerProvider\n'
            'assert isinstance(trace.get_tracer_provider(), ProxyTracerProvider)\n'
            'import pixeltable.otel._sdk as sdk\n'
            'assert sdk._state.initialized\n'
            'assert not sdk._state.owns_tracer_provider and sdk._state.tracer_provider is None\n'
            'from pixeltable.otel import PixeltableInstrumentor\n'
            'assert PixeltableInstrumentor().is_instrumented\n',
            {'PIXELTABLE_OTEL': '1'},
        )

    @pytest.mark.skipif(not _OTEL_INSTALLED, reason='otel extra not installed')
    @pytest.mark.parametrize(
        'env_overrides',
        [{'PIXELTABLE_OTEL': '0'}, {'PIXELTABLE_OTEL': '1', 'OTEL_SDK_DISABLED': 'true'}],
        ids=['pixeltable_otel', 'otel_sdk_disabled'],
    )
    def test_opt_out(self, init_env: None, env_overrides: dict[str, str]) -> None:
        # the opt-out gates precede the installed-ness probe, so nothing opentelemetry gets imported at all
        _run_py(
            'import pixeltable as pxt\n'
            'pxt.init()\n'
            'import sys\n'
            "assert 'pixeltable.otel' not in sys.modules\n"
            "assert not any(m == 'opentelemetry' or m.startswith('opentelemetry.') for m in sys.modules), "
            "[m for m in sys.modules if m.startswith('opentelemetry')]\n",
            env_overrides,
        )

    @pytest.mark.skipif(not _OTEL_INSTALLED, reason='otel extra not installed')
    def test_respect_existing_sdk(self, init_env: None) -> None:
        _run_py(
            'from opentelemetry import trace\n'
            'from opentelemetry.sdk.trace import TracerProvider\n'
            'my_tp = TracerProvider()\n'
            'trace.set_tracer_provider(my_tp)\n'
            'import pixeltable as pxt\n'
            'pxt.init()\n'
            'assert trace.get_tracer_provider() is my_tp\n'
            'import pixeltable.otel._sdk as sdk\n'
            'assert sdk._state.initialized and not sdk._state.owns_tracer_provider\n'
            'from pixeltable.otel import PixeltableInstrumentor\n'
            'assert PixeltableInstrumentor().is_instrumented\n',
            {'PIXELTABLE_OTEL': '1'},
        )

    def test_not_installed_simulation(self, init_env: None) -> None:
        # simulate the extra being absent: the gate declines and core comes up untraced
        _run_py(
            'import importlib.util\n'
            'real_find_spec = importlib.util.find_spec\n'
            'def fake_find_spec(name, *args, **kwargs):\n'
            "    if name.startswith('opentelemetry'):\n"
            '        return None\n'
            '    return real_find_spec(name, *args, **kwargs)\n'
            'importlib.util.find_spec = fake_find_spec\n'
            'import pixeltable as pxt\n'
            'pxt.init()\n'
            'import sys\n'
            "assert 'pixeltable.otel' not in sys.modules\n"
            "assert not any(m.startswith('opentelemetry') for m in sys.modules)\n",
            {'PIXELTABLE_OTEL': '1'},
        )

    @pytest.mark.skipif(not _OTEL_INSTALLED, reason='otel extra not installed')
    def test_failure_isolation(self, init_env: None) -> None:
        # a broken otel config must not block pixeltable bring-up
        _run_py(
            'import pixeltable as pxt\n'
            'pxt.init()\n'
            "t = pxt.create_table('otel_failure_isolation', {'a': pxt.Int}, if_exists='replace')\n"
            "t.insert([{'a': 1}])\n"
            "pxt.drop_table('otel_failure_isolation')\n",
            {'PIXELTABLE_OTEL': '1', 'OTEL_ENDPOINT': '::not a url::'},
        )

    @pytest.mark.skipif(not _OTEL_INSTALLED, reason='otel extra not installed')
    def test_env_reinit(self, init_env: None) -> None:
        _run_py(
            'import pixeltable as pxt\n'
            'pxt.init()\n'
            'from opentelemetry import trace\n'
            'tp = trace.get_tracer_provider()\n'
            'from pixeltable.env import Env\n'
            'Env._init_env()\n'
            'assert trace.get_tracer_provider() is tp\n'
            'from pixeltable.otel import PixeltableInstrumentor\n'
            'assert PixeltableInstrumentor().is_instrumented\n',
            {
                'PIXELTABLE_OTEL': '1',
                'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://127.0.0.1:9',
                'OTEL_EXPORTER_OTLP_TIMEOUT': '1',
            },
        )

    @pytest.mark.skipif(not _OTEL_INSTALLED, reason='otel extra not installed')
    def test_manual_init(self, init_env: None) -> None:
        # manual init() before the first operation; later auto-init is a no-op, second init() raises
        _run_py(
            'import pixeltable as pxt\n'
            'import pixeltable.otel\n'
            "pixeltable.otel.init(endpoint='http://127.0.0.1:9')\n"
            'pxt.init()\n'
            'import pixeltable.otel._sdk as sdk\n'
            'assert sdk._state.user_initialized\n'
            'try:\n'
            '    pixeltable.otel.init()\n'
            "    raise AssertionError('expected init() to raise')\n"
            'except pxt.exceptions.RequestError:\n'
            '    pass\n',
            {'PIXELTABLE_OTEL': '1', 'OTEL_EXPORTER_OTLP_TIMEOUT': '1'},
        )

    @pytest.mark.skipif(not _OTEL_INSTALLED, reason='otel extra not installed')
    def test_protocol_grpc(self, init_env: None) -> None:
        # grpc transport: the span exporter resolves from the grpc OTLP exporter package
        _run_py(
            'import pixeltable as pxt\n'
            'pxt.init()\n'
            'from opentelemetry import trace\n'
            'from opentelemetry.sdk.trace import TracerProvider\n'
            'tp = trace.get_tracer_provider()\n'
            'assert isinstance(tp, TracerProvider), type(tp)\n'
            'import pixeltable.otel._sdk as sdk\n'
            'assert sdk._state.owns_tracer_provider\n'
            'exporter = tp._active_span_processor._span_processors[0].span_exporter\n'
            "assert type(exporter).__module__.startswith('opentelemetry.exporter.otlp.proto.grpc'), "
            'type(exporter).__module__\n',
            {
                'PIXELTABLE_OTEL': '1',
                'OTEL_EXPORTER_OTLP_PROTOCOL': 'grpc',
                'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://127.0.0.1:4317',
                'OTEL_EXPORTER_OTLP_TIMEOUT': '1',
            },
        )


@pytest.mark.skipif(not _OTEL_INSTALLED, reason='otel extra not installed')
class TestProtocolSelection:
    """Unit tests for the OTLP transport selection (pure function, no providers touched)."""

    def test_default_is_http(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pixeltable.otel._sdk import _use_grpc

        monkeypatch.delenv('OTEL_EXPORTER_OTLP_PROTOCOL', raising=False)
        assert _use_grpc(None) is False
        assert _use_grpc('http/protobuf') is False

    def test_grpc_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pixeltable.otel._sdk import _use_grpc

        monkeypatch.delenv('OTEL_EXPORTER_OTLP_PROTOCOL', raising=False)
        assert _use_grpc('grpc') is True

    def test_env_overrides_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pixeltable.otel._sdk import _use_grpc

        monkeypatch.setenv('OTEL_EXPORTER_OTLP_PROTOCOL', 'grpc')
        assert _use_grpc('http/protobuf') is True


@pytest.mark.skipif(not _OTEL_INSTALLED, reason='otel extra not installed')
class TestOtelBridge:
    """In-process bridge correctness with in-memory exporters (no global providers touched)."""

    @pytest.fixture
    def instrumented(self) -> Iterator[tuple[Any, Any]]:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from pixeltable.otel import PixeltableInstrumentor

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

    def test_log_correlation(self, instrumented: tuple[Any, Any]) -> None:
        from opentelemetry import trace

        from pixeltable import hooks

        span_handle = hooks.span_start('pixeltable.insert', set_current=True)
        try:
            record = logging.getLogger('pixeltable').makeRecord('pixeltable', logging.INFO, __file__, 1, 'm', (), None)
            from pixeltable.otel.logs import TRACE_CONTEXT_FILTER

            TRACE_CONTEXT_FILTER.filter(record)
            current = trace.get_current_span().get_span_context()
            assert current.is_valid
            assert record.__dict__['otelTraceID'] == trace.format_trace_id(current.trace_id)
            assert record.__dict__['otelSpanID'] == trace.format_span_id(current.span_id)
        finally:
            hooks.span_end(span_handle)

    def test_token_extraction(self, instrumented: tuple[Any, Any]) -> None:
        from pixeltable import hooks

        _, metric_reader = instrumented
        result = {'model': 'gpt-4o-mini', 'usage': {'prompt_tokens': 10, 'completion_tokens': 3}}
        hooks.emit(
            'udf.call', attrs={'pxt.udf': 'openai.chat_completions', 'duration_s': 0.5, 'count': 1, '_result': result}
        )
        points = self._metric_points(metric_reader, 'pixeltable.udf.tokens')
        by_type = {p.attributes['type']: p.value for p in points}
        assert by_type == {'input': 10, 'output': 3}
        assert all(p.attributes['model'] == 'gpt-4o-mini' for p in points)
