import asyncio
import json
import logging
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator

import opentelemetry.instrumentation.pixeltable as pxt_otel
import pytest
from opentelemetry import metrics as otel_metrics, trace
from opentelemetry.instrumentation.pixeltable import PixeltableInstrumentor, _sdk
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import ProxyTracerProvider, StatusCode

import pixeltable as pxt
from pixeltable import telemetry, telemetry_schemas
from pixeltable.catalog import IfExistsParam, Path as CatalogPath
from pixeltable.service import proxy_daemon, proxy_dispatch
from pixeltable.service.proxy_protocol import InlinePartSink, ProxyRequest, encode_body, serialize_args
from pixeltable.serving._app import instrument_app
from pixeltable.telemetry import SubscriberRegistry


@pxt.udf
def fail_on_marker(s: str) -> str:
    if s == 'fail':
        raise ValueError('deliberate failure')
    return s.upper()


@pxt.udf
async def mock_llm(s: str) -> dict:
    await asyncio.sleep(0)
    return {'choice': s}


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


# Core pixeltable must not depend on opentelemetry: it imports and runs with the extra absent, and nothing
# in core imports it at module load. The serving worker configures tracing in its own process before it
# serves, so it imports the instrumentation lazily, behind require_package().
_LAZY_OTEL_IMPORTS_IN_CORE: frozenset[str] = frozenset({'serving/_app.py'})


def test_no_otel_imports_in_core() -> None:
    pkg_root = Path(pxt.__file__).parent
    module_scope: list[str] = []
    lazy: set[str] = set()
    for path in pkg_root.rglob('*.py'):
        source = path.read_text()
        rel = path.relative_to(pkg_root).as_posix()
        if re.search(r'^(import opentelemetry|from opentelemetry)', source, flags=re.MULTILINE):
            module_scope.append(rel)
        if re.search(r'^\s+(import opentelemetry|from opentelemetry)', source, flags=re.MULTILINE):
            lazy.add(rel)
    assert module_scope == []
    assert lazy == _LAZY_OTEL_IMPORTS_IN_CORE


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
    # Env stops these loggers from propagating to the root logger, so each one carries the export handler
    for logger_name in ('pixeltable', 'sqlalchemy.engine', 'uvicorn'):
        handlers = logging.getLogger(logger_name).handlers
        assert any(type(handler).__name__ == 'LoggingHandler' for handler in handlers), logger_name


def test_logs_can_be_enabled_explicitly(tmp_path: Path) -> None:
    _run_isolated(
        check_logs_can_be_enabled_explicitly,
        {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://127.0.0.1:9', 'OTEL_EXPORTER_OTLP_TIMEOUT': '1'},
        tmp_path,
    )


def check_inert_without_endpoint() -> None:
    for k in list(os.environ):
        if k.startswith('OTEL_'):
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


def check_metrics_off_skips_app_meter_provider() -> None:
    reader = InMemoryMetricReader()
    otel_metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
    pxt_otel.init(metrics=False)
    telemetry_schemas.rows_written.add(1, table='t')
    metrics_data = reader.get_metrics_data()
    assert metrics_data is None or metrics_data.resource_metrics == []


def test_metrics_off_skips_app_meter_provider(tmp_path: Path) -> None:
    # init(metrics=False) records nothing, even into a meter provider the application installed
    _run_isolated(check_metrics_off_skips_app_meter_provider, {}, tmp_path)


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


def check_invalid_config_leaves_init_retryable() -> None:
    # an unknown span level, and a span_dump file that cannot be opened
    for invalid in ({'span_level': 'verbose'}, {'span_dump': '/nonexistent-dir/spans.jsonl'}):
        try:
            pxt_otel.init(**invalid)  # type: ignore[arg-type]
            raise AssertionError('expected init() to raise')
        except pxt.exceptions.RequestError as e:
            assert e.error_code is pxt.exceptions.ErrorCode.INVALID_CONFIGURATION
        assert not _sdk._state.initialized
    pxt_otel.init(span_level='debug')
    assert _sdk._state.initialized
    assert SubscriberRegistry.get()._span_level == telemetry.DEBUG


def test_invalid_config_leaves_init_retryable(tmp_path: Path) -> None:
    _run_isolated(check_invalid_config_leaves_init_retryable, {'OTEL_EXPORTER_OTLP_TIMEOUT': '1'}, tmp_path)


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


def check_span_dump_without_endpoint() -> None:
    for k in list(os.environ):
        if k.startswith('OTEL_') and k != 'OTEL_SPAN_DUMP':
            del os.environ[k]
    pxt_otel.init()
    assert isinstance(trace.get_tracer_provider(), TracerProvider)
    assert _sdk._state.owns_tracer_provider
    parent = telemetry.span_start('pixeltable.op', set_current=True)
    telemetry.span_end(telemetry.span_start('pixeltable.udf.foo', set_current=True))
    telemetry.span_end(parent)
    spans = {}
    for line in Path(os.environ['OTEL_SPAN_DUMP']).read_text(encoding='utf-8').splitlines():
        span = json.loads(line)
        spans[span['name']] = span
    assert set(spans) == {'pixeltable.op', 'pixeltable.udf.foo'}
    assert spans['pixeltable.udf.foo']['parent_id'] == spans['pixeltable.op']['context']['span_id']


def test_span_dump_without_endpoint(tmp_path: Path) -> None:
    # a span dump alone activates export (no longer inert): spans land in the file as JSON lines,
    # written synchronously on span end
    _run_isolated(check_span_dump_without_endpoint, {'OTEL_SPAN_DUMP': str(tmp_path / 'spans.jsonl')}, tmp_path)


def check_span_dump_composes_with_otlp() -> None:
    pxt_otel.init()
    tp = trace.get_tracer_provider()
    processors = tp._active_span_processor._span_processors  # type: ignore[attr-defined]
    assert {type(p).__name__ for p in processors} == {'BatchSpanProcessor', 'SimpleSpanProcessor'}


def test_span_dump_composes_with_otlp(tmp_path: Path) -> None:
    # endpoint + span dump: both processors attach to the same provider
    _run_isolated(
        check_span_dump_composes_with_otlp,
        {
            'OTEL_SPAN_DUMP': str(tmp_path / 'spans.jsonl'),
            'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://127.0.0.1:9',
            'OTEL_EXPORTER_OTLP_TIMEOUT': '1',
        },
        tmp_path,
    )


def check_endpoint_from_database_entry() -> None:
    for k in list(os.environ):
        if k.startswith(('OTEL_EXPORTER_OTLP_ENDPOINT', 'OTEL_EXPORTER_OTLP_PROTOCOL')):
            del os.environ[k]
    Path(os.environ['PIXELTABLE_CONFIG']).write_text(
        "[pixeltable]\nfile_cache_size_g = 1.0\n\n[[pixeltable.database]]\ndb_exporter_otlp_endpoint = 'http://127.0.0.1:9'\n",
        encoding='utf-8',
    )
    pxt_otel.init()
    tp = trace.get_tracer_provider()
    exporter = tp._active_span_processor._span_processors[0].span_exporter  # type: ignore[attr-defined]
    assert exporter._endpoint == 'http://127.0.0.1:9/v1/traces', exporter._endpoint


def test_endpoint_from_database_entry(tmp_path: Path) -> None:
    # the entry for the process's database sets the endpoint, in place of the [otel] section or the env var
    _run_isolated(
        check_endpoint_from_database_entry,
        {'PIXELTABLE_CONFIG': str(tmp_path / 'config.toml'), 'OTEL_EXPORTER_OTLP_TIMEOUT': '1'},
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
    def span_exporter(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        # the provider that init() would have built, which instrument_app() wires into the http instrumentation
        monkeypatch.setattr(_sdk._state, 'tracer_provider', tracer_provider)
        instrumentor = PixeltableInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        yield span_exporter
        instrumentor.uninstrument()

    def test_span_capture(self, span_exporter: Any) -> None:
        handle = telemetry.span_start('pixeltable.insert', set_current=True, attrs={'pxt.table_id': 't1'})
        telemetry.add_attrs(handle, num_rows=5, updated_cols=['a', 'b'])
        telemetry.span_end(handle)
        (span,) = span_exporter.get_finished_spans()
        assert span.name == 'pixeltable.insert'
        # start attrs and add_attrs() end attrs both land on the exported span
        assert span.attributes['pxt.table_id'] == 't1'
        assert span.attributes['pxt.num_rows'] == 5
        assert tuple(span.attributes['pxt.updated_cols']) == ('a', 'b')

    def test_span_nesting(self, span_exporter: Any) -> None:
        parent = telemetry.span_start('pixeltable.op', set_current=True)
        child = telemetry.span_start('pixeltable.udf.foo', set_current=True)
        telemetry.span_end(child)
        telemetry.span_end(parent)
        spans = {s.name: s for s in span_exporter.get_finished_spans()}
        assert spans['pixeltable.udf.foo'].parent is not None
        assert spans['pixeltable.udf.foo'].parent.span_id == spans['pixeltable.op'].context.span_id

    def test_span_event(self, span_exporter: Any) -> None:
        handle = telemetry.span_start('pixeltable.insert', set_current=True)
        telemetry.emit(handle, 'pixeltable.cell.error', column='c', count=2, skipped=None)
        telemetry.span_end(handle)
        (span,) = span_exporter.get_finished_spans()
        (event,) = span.events
        assert event.name == 'pixeltable.cell.error'
        assert dict(event.attributes) == {'pxt.column': 'c', 'pxt.count': 2}

    def test_daemon_request_spans(self, span_exporter: Any) -> None:
        pytest.importorskip('opentelemetry.instrumentation.fastapi')
        from fastapi.testclient import TestClient

        app = instrument_app(proxy_daemon._build_app())
        args = {'path': CatalogPath.parse('otel_rpc'), 'if_exists': IfExistsParam.ERROR, 'parents': False}
        request = ProxyRequest(
            class_name='CatalogBase', method='create_dir', args=serialize_args(args, InlinePartSink())
        )
        body = encode_body(request.model_dump_json().encode(), [])
        client_traceparent = f'00-{"a" * 32}-{"b" * 16}-01'
        sent_at = time.time()
        try:
            with TestClient(app) as client:
                assert client.get('/health').status_code == 200
                assert client.get('/health', headers={'x-pxt-gateway-time': f'{sent_at} 0.25'}).status_code == 200
                # the first call creates the directory and the other five fail on it
                assert client.post('/rpc', content=body).status_code == 200
                assert client.post('/rpc', content=body).status_code == 200
                gateway_headers = {'x-pxt-gateway-time': f'{sent_at} 0.25', 'traceparent': client_traceparent}
                assert client.post('/rpc', content=body, headers=gateway_headers).status_code == 200
                # a malformed gateway header, or one far from this pod's clock, counts as no header
                for timing in ('nope', '1 0', f'{sent_at + 3600} 0'):
                    assert client.post('/rpc', content=body, headers={'x-pxt-gateway-time': timing}).status_code == 200
        finally:
            pxt.drop_dir('otel_rpc', force=True)

        spans = span_exporter.get_finished_spans()
        span_ids = {s.context.span_id: s for s in spans}
        # the health probes stay out of the traces, also when they come through the gateway
        assert not any('/health' in s.name for s in spans)
        assert [s.name for s in spans].count('pixeltable.gateway') == 1
        # a request exports no span per ASGI message
        assert not any(s.name.endswith((' http send', ' http receive')) for s in spans)
        rpc_spans = sorted((s for s in spans if s.name == 'pixeltable.rpc'), key=lambda s: s.start_time)
        assert [s.attributes['pxt.method'] for s in rpc_spans] == ['CatalogBase.create_dir'] * 6
        # a failed call marks its span, so that tail sampling keeps the trace
        assert [s.status.status_code for s in rpc_spans] == [StatusCode.UNSET] + [StatusCode.ERROR] * 5
        request_spans = [span_ids[s.parent.span_id] for s in rpc_spans]
        assert {s.name for s in request_spans} == {'POST /rpc'}
        # catalog work of the call nests under its rpc span
        assert any(s.parent is not None and s.parent.span_id == rpc_spans[0].context.span_id for s in spans)
        # without a valid gateway header, the request span is the root of its trace
        assert all(request_spans[i].parent is None for i in (0, 1, 3, 4, 5))
        # with it, the trace starts when the gateway received the request, and the client's traceparent is ignored
        gateway_span = span_ids[request_spans[2].parent.span_id]
        assert gateway_span.name == 'pixeltable.gateway' and gateway_span.parent is None
        assert gateway_span.start_time == int((sent_at - 0.25) * 1e9)
        assert gateway_span.end_time >= request_spans[2].end_time
        assert f'{gateway_span.context.trace_id:032x}' != 'a' * 32

    def test_serving_route_spans(self, span_exporter: Any) -> None:
        pytest.importorskip('opentelemetry.instrumentation.fastapi')
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        pxt.create_dir('otel_serving', if_exists='replace_force')
        t = pxt.create_table('otel_serving.docs', {'id': pxt.Int})
        t.insert([{'id': 1}])

        @pxt.query
        def lookup(min_id: int) -> pxt.Query:
            return t.where(t.id >= min_id).select(t.id)

        router = FastAPIRouter()
        router.add_query_route(path='/lookup', query=lookup)
        router.add_insert_route(t, path='/bg', background=True)
        app = FastAPI()
        app.include_router(router)
        try:
            with TestClient(instrument_app(app)) as client:
                assert client.post('/lookup', json={'min_id': 1}).status_code == 200
                job = client.post('/bg', json={'id': 2}).json()
                deadline = time.time() + 30.0
                while client.get(job['job_url']).json()['status'] == 'pending' and time.time() < deadline:
                    time.sleep(0.05)
        finally:
            pxt.drop_dir('otel_serving', force=True)

        spans = span_exporter.get_finished_spans()
        by_id = {s.context.span_id: s for s in spans}
        by_name = {s.name: s for s in spans}
        route_spans = {s.attributes['pxt.route']: s for s in spans if s.name == 'pixeltable.route'}
        # each route's work runs in a route span under its request span
        assert set(route_spans) == {'POST /lookup', 'POST /bg'}
        assert by_id[route_spans['POST /lookup'].parent.span_id].name == 'POST /lookup'
        # the query route's collect nests under its route span and carries the table and row count
        collect = by_name['pixeltable.collect']
        assert collect.parent.span_id == route_spans['POST /lookup'].context.span_id
        assert collect.attributes['pxt.tables'] == ('docs',)
        assert collect.attributes['pxt.rows'] == 1
        # the insert route checks the table's schema before the insert, and that check's catalog work nests under it
        insert = by_name['pixeltable.insert']
        check = by_name['pixeltable.route.check_schema']
        bg_route = route_spans['POST /bg']
        assert insert.parent.span_id == bg_route.context.span_id
        assert check.parent.span_id == bg_route.context.span_id
        assert check.end_time <= insert.start_time
        assert any(s.parent is not None and s.parent.span_id == check.context.span_id for s in spans)
        # the background insert runs on a worker thread, in the trace of the request that queued it
        assert insert.context.trace_id == by_name['POST /bg'].context.trace_id

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
        # a declared bucket-boundary advisory reaches the SDK aggregation
        hb = telemetry.histogram('pixeltable.test.latency_bounded', unit='s', boundaries=(0.1, 1.0, 10.0))
        hb.record(0.5, udf='f')
        metric = self._metric(metric_reader, 'pixeltable.test.latency_bounded')
        (point,) = metric.data.data_points
        assert list(point.explicit_bounds) == [0.1, 1.0, 10.0]

    def test_daemon_request_metrics(self, metric_reader: Any) -> None:
        args = {'path': CatalogPath.parse('otel_rpc'), 'if_exists': IfExistsParam.ERROR, 'parents': False}
        request = ProxyRequest(
            class_name='CatalogBase', method='create_dir', args=serialize_args(args, InlinePartSink())
        )
        try:
            proxy_dispatch.handle(request.model_dump_json(), [])
            # a client with another protocol version gets an error response, which counts as a failed call
            proxy_dispatch.handle(request.model_copy(update={'protocol_version': 0}).model_dump_json(), [])
            # a method that the client makes up is recorded as 'unsupported'
            proxy_dispatch.handle(request.model_copy(update={'method': 'no_such_method'}).model_dump_json(), [])
        finally:
            pxt.drop_dir('otel_rpc', force=True)
        metric = self._metric(metric_reader, 'pixeltable.rpc.duration')
        counts = {(p.attributes['pxt.method'], p.attributes['pxt.error']): p.count for p in metric.data.data_points}
        assert counts == {
            ('CatalogBase.create_dir', False): 1,
            ('CatalogBase.create_dir', True): 1,
            ('unsupported', True): 1,
        }

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
            # 4 calls on insert, 1 of them failed; not recomputed by the update ('checked' depends on 'b')
            assert point('pixeltable.udf.calls', udf='fail_on_marker').value == 4

            latency = point('pixeltable.udf.latency', udf='casefold')
            assert latency.count == 8
            assert latency.sum >= 0
        finally:
            pxt.drop_dir('otel_metrics_smoke', force=True)

    def test_token_usage_export(self, metric_reader: Any) -> None:
        # record_token_usage() is called by the instrumented provider UDFs (openai/anthropic/gemini)
        # with their response's usage dict; exercised directly here to avoid a remote API call
        telemetry_schemas.record_token_usage(
            'pixeltable.functions.openai.chat_completions',
            'gpt-4o-mini',
            {'prompt_tokens': 7, 'completion_tokens': 3},
            'prompt_tokens',
            'completion_tokens',
        )
        in_point = self._metric(metric_reader, 'pixeltable.udf.input_tokens').data.data_points[0]
        out_point = self._metric(metric_reader, 'pixeltable.udf.output_tokens').data.data_points[0]
        assert in_point.value == 7
        assert out_point.value == 3
        assert dict(in_point.attributes) == {
            'pxt.udf': 'chat_completions',
            'pxt.udf_path': 'pixeltable.functions.openai.chat_completions',
            'pxt.model': 'gpt-4o-mini',
        }

    def test_cell_error_event(self, span_exporter: Any) -> None:
        # at DEBUG, cell.error events attach to the per-row spans; at INFO (no row spans) they fall
        # back to the ambient span, the insert's pixeltable.store.insert_rows
        telemetry.set_span_level(telemetry.DEBUG)
        pxt.create_dir('otel_event_smoke', if_exists='replace_force')
        try:
            t = pxt.create_table('otel_event_smoke.tbl', {'b': pxt.String})
            t.add_computed_column(checked=fail_on_marker(t.b))
            status = t.insert([{'b': 'fail'}, {'b': 'ok'}], on_error='ignore')
            assert status.num_excs > 0
            spans = span_exporter.get_finished_spans()
            events = [e for s in spans if s.name == 'pixeltable.row' for e in s.events]
            (event,) = events  # one failing row -> one event, on its row span
            assert event.name == 'pixeltable.cell.error'
            assert event.attributes['pxt.column'] == 'checked'
            assert event.attributes['pxt.error'] == 'ValueError'
            insert_span = next(s for s in spans if s.name == 'pixeltable.insert')
            assert insert_span.events == ()

            telemetry.set_span_level(telemetry.INFO)
            status = t.insert([{'b': 'fail'}], on_error='ignore')
            assert status.num_excs > 0
            spans = span_exporter.get_finished_spans()
            info_insert_rows = [s for s in spans if s.name == 'pixeltable.store.insert_rows'][-1]
            (event,) = info_insert_rows.events
            assert event.name == 'pixeltable.cell.error'
            assert event.attributes['pxt.column'] == 'checked'
        finally:
            pxt.drop_dir('otel_event_smoke', force=True)
            telemetry.set_span_level(telemetry.INFO)

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
