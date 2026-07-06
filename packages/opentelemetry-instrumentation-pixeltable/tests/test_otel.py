import logging
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator

import opentelemetry.instrumentation.pixeltable as pxt_otel
import pytest
from opentelemetry import trace
from opentelemetry.instrumentation.pixeltable import PixeltableInstrumentor, _sdk
from opentelemetry.instrumentation.pixeltable._sdk import _use_grpc
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import ProxyTracerProvider

import pixeltable as pxt
from pixeltable import hooks


def _set_env(env: dict[str, str]) -> None:
    os.environ.update(env)


def _run_isolated(fn: Callable[[], None], env_overrides: dict[str, str], tmp_home: Path) -> None:
    """Run fn in a spawned process: init() reads pixeltable config, and global OTEL providers are set-once
    per process.

    fn must be module-level: spawn pickles it by qualified name for the child to re-import, so nested
    functions can't be used. Env is applied via the initializer, before the child imports fn's module.
    """
    env = {'PIXELTABLE_HOME': str(tmp_home), **env_overrides}
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_set_env, initargs=(env,)) as pool:
        pool.submit(fn).result()


def test_no_otel_imports_in_core() -> None:
    # core pixeltable must never import opentelemetry; instrumentation lives only in this package
    pkg_root = Path(pxt.__file__).parent
    offenders = [
        str(path)
        for path in pkg_root.rglob('*.py')
        if re.search(r'^\s*(import opentelemetry|from opentelemetry)', path.read_text(), flags=re.MULTILINE)
    ]
    assert offenders == []


def check_builds_tracer_provider() -> None:
    pxt_otel.init()
    tp = trace.get_tracer_provider()
    assert isinstance(tp, TracerProvider), type(tp)
    assert tp.resource.attributes['service.name'] == 'pixeltable'
    assert _sdk._state.initialized and _sdk._state.owns_tracer_provider and _sdk._state.owns_meter_provider
    assert PixeltableInstrumentor().is_instrumented_by_opentelemetry


def test_builds_tracer_provider(tmp_path: Path) -> None:
    _run_isolated(
        check_builds_tracer_provider,
        {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://127.0.0.1:9', 'OTEL_EXPORTER_OTLP_TIMEOUT': '1'},
        tmp_path,
    )


def check_inert_without_endpoint() -> None:
    for k in [k for k in os.environ if k.startswith('OTEL_')]:
        del os.environ[k]
    pxt_otel.init()
    assert isinstance(trace.get_tracer_provider(), ProxyTracerProvider)
    assert _sdk._state.initialized
    assert not _sdk._state.owns_tracer_provider and _sdk._state.tracer_provider is None
    assert PixeltableInstrumentor().is_instrumented_by_opentelemetry


def test_inert_without_endpoint(tmp_path: Path) -> None:
    # no endpoint configured and no app-owned provider: the bridge attaches but nothing is exported
    _run_isolated(check_inert_without_endpoint, {}, tmp_path)


def check_respects_existing_sdk() -> None:
    my_tp = TracerProvider()
    trace.set_tracer_provider(my_tp)
    pxt_otel.init()
    assert trace.get_tracer_provider() is my_tp
    assert _sdk._state.initialized and not _sdk._state.owns_tracer_provider


def test_respects_existing_sdk(tmp_path: Path) -> None:
    _run_isolated(check_respects_existing_sdk, {}, tmp_path)


def check_double_init_raises() -> None:
    pxt_otel.init()
    try:
        pxt_otel.init()
        raise AssertionError('expected init() to raise')
    except pxt.exceptions.RequestError:
        pass


def test_double_init_raises(tmp_path: Path) -> None:
    _run_isolated(check_double_init_raises, {'OTEL_EXPORTER_OTLP_TIMEOUT': '1'}, tmp_path)


def check_protocol_grpc() -> None:
    pxt_otel.init()
    tp = trace.get_tracer_provider()
    exporter = tp._active_span_processor._span_processors[0].span_exporter  # type: ignore[attr-defined]
    assert type(exporter).__module__.startswith('opentelemetry.exporter.otlp.proto.grpc'), type(exporter).__module__


def test_protocol_grpc(tmp_path: Path) -> None:
    _run_isolated(
        check_protocol_grpc,
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
        monkeypatch.delenv('OTEL_EXPORTER_OTLP_PROTOCOL', raising=False)
        assert _use_grpc(None) is False
        assert _use_grpc('http/protobuf') is False

    def test_grpc_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('OTEL_EXPORTER_OTLP_PROTOCOL', raising=False)
        assert _use_grpc('grpc') is True

    def test_env_overrides_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('OTEL_EXPORTER_OTLP_PROTOCOL', 'grpc')
        assert _use_grpc('http/protobuf') is True


class TestBridge:
    """In-process bridge correctness with in-memory exporters (no global providers touched)."""

    @pytest.fixture
    def span_exporter(self) -> Iterator[Any]:
        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        instrumentor = PixeltableInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        yield span_exporter
        instrumentor.uninstrument()

    def test_span_capture(self, span_exporter: Any) -> None:
        hooks.span_end(hooks.span_start('pixeltable.insert', set_current=True))
        assert [s.name for s in span_exporter.get_finished_spans()] == ['pixeltable.insert']

    def test_span_nesting(self, span_exporter: Any) -> None:
        parent = hooks.span_start('pixeltable.op', set_current=True)
        child = hooks.span_start('pixeltable.udf.foo', set_current=True)
        hooks.span_end(child)
        hooks.span_end(parent)
        spans = {s.name: s for s in span_exporter.get_finished_spans()}
        assert spans['pixeltable.udf.foo'].parent is not None
        assert spans['pixeltable.udf.foo'].parent.span_id == spans['pixeltable.op'].context.span_id

    def test_uninstrument_stops_emission(self) -> None:
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

    def test_log_correlation(self, span_exporter: Any) -> None:
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
