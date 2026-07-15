import asyncio
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
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import ProxyTracerProvider

import pixeltable as pxt
from pixeltable import telemetry
from pixeltable.telemetry import SubscriberRegistry


@pxt.udf
def fail_on_marker(s: str) -> str:
    if s == 'fail':
        raise ValueError('deliberate failure')
    return s.upper()


@pxt.udf
async def mock_llm(s: str) -> dict:
    await asyncio.sleep(0)
    return {'choice': s, 'usage': {'prompt_tokens': 7, 'completion_tokens': 3}}


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


def check_logs_require_explicit_opt_in() -> None:
    pxt_otel.init()
    assert _sdk._state.logger_provider is None


def test_logs_require_explicit_opt_in(tmp_path: Path) -> None:
    _run_isolated(
        check_logs_require_explicit_opt_in,
        {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://127.0.0.1:9', 'OTEL_EXPORTER_OTLP_TIMEOUT': '1'},
        tmp_path,
    )


def check_logs_can_be_enabled_explicitly() -> None:
    pxt_otel.init(logs=True)
    assert _sdk._state.logger_provider is not None


def test_logs_can_be_enabled_explicitly(tmp_path: Path) -> None:
    _run_isolated(
        check_logs_can_be_enabled_explicitly,
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


def check_span_level_kwarg_overrides_config() -> None:
    pxt_otel.init(span_level='debug')
    assert SubscriberRegistry.get()._span_level == telemetry.DEBUG


def test_span_level_kwarg_overrides_config(tmp_path: Path) -> None:
    # the init() argument wins over the config-derived value
    _run_isolated(
        check_span_level_kwarg_overrides_config,
        {'OTEL_SPAN_LEVEL': 'trace', 'OTEL_EXPORTER_OTLP_TIMEOUT': '1'},
        tmp_path,
    )


def check_invalid_span_level_leaves_init_retryable() -> None:
    try:
        pxt_otel.init(span_level='verbose')  # type: ignore[arg-type]
        raise AssertionError('expected init() to raise')
    except pxt.exceptions.RequestError as e:
        assert e.error_code is pxt.exceptions.ErrorCode.INVALID_CONFIGURATION
    assert not _sdk._state.initialized
    pxt_otel.init(span_level='debug')
    assert _sdk._state.initialized
    assert SubscriberRegistry.get()._span_level == telemetry.DEBUG


def test_invalid_span_level_leaves_init_retryable(tmp_path: Path) -> None:
    _run_isolated(check_invalid_span_level_leaves_init_retryable, {'OTEL_EXPORTER_OTLP_TIMEOUT': '1'}, tmp_path)


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


def test_standard_header_parsing() -> None:
    assert _sdk._parse_headers('Authorization=Basic%20abc%2Cdef,X-Test=a%20b') == {
        'authorization': 'Basic abc,def',
        'x-test': 'a b',
    }


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
        telemetry.span_end(telemetry.span_start('pixeltable.insert', set_current=True))
        assert [s.name for s in span_exporter.get_finished_spans()] == ['pixeltable.insert']

    def test_span_nesting(self, span_exporter: Any) -> None:
        parent = telemetry.span_start('pixeltable.op', set_current=True)
        child = telemetry.span_start('pixeltable.udf.foo', set_current=True)
        telemetry.span_end(child)
        telemetry.span_end(parent)
        spans = {s.name: s for s in span_exporter.get_finished_spans()}
        assert spans['pixeltable.udf.foo'].parent is not None
        assert spans['pixeltable.udf.foo'].parent.span_id == spans['pixeltable.op'].context.span_id

    @pytest.fixture
    def metric_reader(self) -> Iterator[Any]:
        reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[reader])
        instrumentor = PixeltableInstrumentor()
        instrumentor.instrument(meter_provider=meter_provider)
        yield reader
        instrumentor.uninstrument()
        meter_provider.shutdown()

    def _metric(self, reader: Any, name: str) -> Any:
        data = reader.get_metrics_data()
        all_metrics = [m for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics]
        matches = [m for m in all_metrics if m.name == name]
        assert len(matches) == 1, f'expected exactly one metric named {name!r}, got {len(matches)}'
        return matches[0]

    def test_counter_export(self, metric_reader: Any) -> None:
        c = telemetry.counter('pixeltable.test.rows', unit='{row}')
        c.add(3, table='dir.tbl')
        c.add(5, table='dir.tbl')
        metric = self._metric(metric_reader, 'pixeltable.test.rows')
        assert metric.unit == '{row}'
        (point,) = metric.data.data_points
        assert point.value == 8
        assert dict(point.attributes) == {'pxt.table': 'dir.tbl'}

    def test_histogram_export(self, metric_reader: Any) -> None:
        h = telemetry.histogram('pixeltable.test.latency', unit='s')
        h.record(0.5, udf='f')
        h.record(1.5, udf='f')
        metric = self._metric(metric_reader, 'pixeltable.test.latency')
        (point,) = metric.data.data_points
        assert point.count == 2
        assert point.sum == 2.0
        assert dict(point.attributes) == {'pxt.udf': 'f'}

    def test_metrics_end_to_end(self, metric_reader: Any) -> None:
        pxt.create_dir('otel_metrics_smoke', if_exists='replace_force')
        try:
            t = pxt.create_table('otel_metrics_smoke.tbl', {'a': pxt.String, 'b': pxt.String})
            # casefold: an existing scalar sync UDF without a to_sql translation, so both the insert and
            # the update recompute run through the Python evaluator and the metric counts are deterministic
            t.add_computed_column(folded=t.a.casefold())
            t.add_computed_column(checked=fail_on_marker(t.b))
            t.add_computed_column(llm=mock_llm(t.a))
            rows = [{'a': f'r{i}', 'b': 'fail' if i == 0 else f'r{i}'} for i in range(4)]
            status = t.insert(rows, on_error='ignore')
            assert status.num_excs > 0
            t.update({'a': t.a + '!'})

            def point(name: str, **attrs: str) -> Any:
                metric = self._metric(metric_reader, name)
                pts = [
                    p
                    for p in metric.data.data_points
                    if all(dict(p.attributes).get(f'pxt.{k}') == v for k, v in attrs.items())
                ]
                assert len(pts) == 1, f'{name} {attrs}: expected 1 matching data point, got {len(pts)}'
                return pts[0]

            # 4 rows inserted, then re-inserted as new versions by the update
            rows_written = point('pixeltable.rows.written', table='tbl')
            assert rows_written.value == 8
            assert 'pxt.table_id' in dict(rows_written.attributes)

            assert point('pixeltable.cells.computed', table='tbl').value >= 8

            assert point('pixeltable.cells.errors', table='tbl').value == 1
            assert point('pixeltable.udf.errors', udf='fail_on_marker').value == 1

            # 4 calls on insert + 4 on update (both columns depend on 'a')
            assert point('pixeltable.udf.calls', udf='casefold').value == 8
            assert point('pixeltable.udf.calls', udf='mock_llm').value == 8
            # 4 attempts on insert, 1 failed; not recomputed by the update ('checked' depends on 'b')
            assert point('pixeltable.udf.calls', udf='fail_on_marker').value == 3

            latency = point('pixeltable.udf.latency', udf='casefold')
            assert latency.count == 8
            assert latency.sum >= 0

            assert point('pixeltable.udf.input_tokens', udf='mock_llm').value == 7 * 8
            assert point('pixeltable.udf.output_tokens', udf='mock_llm').value == 3 * 8
        finally:
            pxt.drop_dir('otel_metrics_smoke', force=True)

    def test_uninstrument_stops_emission(self) -> None:
        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        instrumentor = PixeltableInstrumentor()
        prev_factory = logging.getLogRecordFactory()
        instrumentor.instrument(tracer_provider=tracer_provider)
        try:
            telemetry.span_end(telemetry.span_start('pixeltable.op', set_current=True))
            assert len(span_exporter.get_finished_spans()) == 1
        finally:
            instrumentor.uninstrument()
        assert not telemetry.active()
        assert logging.getLogRecordFactory() is prev_factory
        telemetry.span_end(telemetry.span_start('pixeltable.op', set_current=True))
        assert len(span_exporter.get_finished_spans()) == 1  # nothing new after uninstrument

    def test_instrument_span_level(self) -> None:
        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        instrumentor = PixeltableInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, span_level='debug')
        try:
            parent = telemetry.span_start('pixeltable.op', set_current=True)
            telemetry.span_end(telemetry.span_start('pixeltable.row', level=telemetry.DEBUG))
            telemetry.span_end(parent)
            assert {s.name for s in span_exporter.get_finished_spans()} == {'pixeltable.op', 'pixeltable.row'}
        finally:
            instrumentor.uninstrument()
            telemetry.set_span_level(telemetry.INFO)

    def test_instrument_invalid_span_level(self) -> None:
        instrumentor = PixeltableInstrumentor()
        with pytest.raises(pxt.exceptions.RequestError) as exc_info:
            instrumentor.instrument(span_level='verbose')
        assert exc_info.value.error_code is pxt.exceptions.ErrorCode.INVALID_CONFIGURATION
        assert not instrumentor.is_instrumented_by_opentelemetry
        assert not telemetry.active()

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
            span_handle = telemetry.span_start('pixeltable.insert', set_current=True)
            try:
                child_logger.info('correlated message')
                ctx = trace.get_current_span().get_span_context()
                assert ctx.is_valid
            finally:
                telemetry.span_end(span_handle)
            record = next(r for r in records if r.getMessage() == 'correlated message')
            assert record.otelTraceID == trace.format_trace_id(ctx.trace_id)  # type: ignore[attr-defined]
            assert record.otelSpanID == trace.format_span_id(ctx.span_id)  # type: ignore[attr-defined]
        finally:
            pxt_logger.removeHandler(handler)
            pxt_logger.setLevel(old_level)
