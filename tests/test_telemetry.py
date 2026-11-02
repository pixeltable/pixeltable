import asyncio
import threading
from typing import Any

import pytest

import pixeltable.functions as pxtf
from pixeltable import telemetry, telemetry_schemas
from pixeltable.telemetry import SubscriberRegistry

from .utils import RecordingSubscriber


class RaisingSubscriber(telemetry.Subscriber):
    def on_span_start(self, name: str, parent_token: Any, attrs: dict[str, Any] | None, set_current: bool) -> Any:
        raise RuntimeError('on_span_start')

    def on_span_end(self, token: Any, exc: BaseException | None, attrs: dict[str, Any] | None) -> None:
        raise RuntimeError('on_span_end')

    def on_event(self, token: Any, name: str, attrs: dict[str, Any]) -> None:
        raise RuntimeError('on_event')

    def on_counter_add(self, counter: telemetry.Counter, value: int | float, attrs: dict[str, Any]) -> None:
        raise RuntimeError('on_counter_add')

    def on_histogram_record(self, histogram: telemetry.Histogram, value: int | float, attrs: dict[str, Any]) -> None:
        raise RuntimeError('on_histogram_record')

    def capture_context(self) -> Any:
        raise RuntimeError('capture_context')

    def restore_context(self, ctx: Any) -> Any:
        raise RuntimeError('restore_context')

    def exit_context(self, token: Any) -> None:
        raise RuntimeError('exit_context')


class TestHooks:
    def test_inactive_is_noop(self) -> None:
        assert not telemetry.active()
        calls: list[str] = []

        def attrs() -> dict[str, Any]:
            calls.append('attrs')
            return {}

        handle = telemetry.span_start('a', attrs=attrs)
        assert handle is None
        telemetry.span_end(handle)
        telemetry.emit(handle, 'e', x=1)
        assert telemetry.capture_context() is None
        telemetry.restore_context(None)
        telemetry.exit_context(None)
        assert calls == []

    def test_subscribe_unsubscribe(self) -> None:
        s = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(s)
        SubscriberRegistry.get().subscribe(s)  # idempotent
        try:
            assert telemetry.active()
            handle = telemetry.span_start('a', set_current=True)
            telemetry.span_end(handle)
            assert len(s.spans) == 1
        finally:
            SubscriberRegistry.get().unsubscribe(s)
        assert not telemetry.active()
        assert telemetry.span_start('b', set_current=True) is None
        assert len(s.spans) == 1

    def test_counter_and_histogram(self, telemetry_recorder: RecordingSubscriber) -> None:
        c = telemetry.counter('pixeltable.test.rows', unit='{row}')
        h = telemetry.histogram('pixeltable.test.latency', unit='s')
        c.add(5, table='dir.tbl', skipped=None)
        h.record(0.25, udf='f')
        assert telemetry_recorder.counter_adds == [(c, 5, {'pxt.table': 'dir.tbl'})]
        assert telemetry_recorder.histogram_records == [(h, 0.25, {'pxt.udf': 'f'})]

    def test_record_token_usage(self, telemetry_recorder: RecordingSubscriber) -> None:
        # both counts present: one add per token counter, with the udf and model dimensions
        path = 'pixeltable.functions.anthropic.messages'
        telemetry_schemas.record_token_usage(
            path, 'claude', {'input_tokens': 7, 'output_tokens': 3}, 'input_tokens', 'output_tokens'
        )
        dims = {'pxt.udf': 'messages', 'pxt.udf_path': path, 'pxt.model': 'claude'}
        assert telemetry_recorder.counter_adds == [
            (telemetry_schemas.udf_input_tokens, 7, dims),
            (telemetry_schemas.udf_output_tokens, 3, dims),
        ]
        # one count present: only that counter records
        telemetry_schemas.record_token_usage(path, 'claude', {'input_tokens': 5}, 'input_tokens', 'output_tokens')
        assert len(telemetry_recorder.counter_adds) == 3
        # non-dict usage and non-int counts record nothing
        telemetry_schemas.record_token_usage(path, 'claude', None, 'input_tokens', 'output_tokens')
        telemetry_schemas.record_token_usage(path, 'claude', {'input_tokens': 'x'}, 'input_tokens', 'output_tokens')
        assert len(telemetry_recorder.counter_adds) == 3
        # with no ambient span there is nothing to attach the event to
        assert telemetry_recorder.events == []
        # inside a span: the token usage also lands on it as a pixeltable.udf.tokens event
        with telemetry.span('op', set_current=True):
            telemetry_schemas.record_token_usage(
                path, 'claude', {'input_tokens': 7, 'output_tokens': 3}, 'input_tokens', 'output_tokens'
            )
        assert telemetry_recorder.events == [
            (
                telemetry_recorder.find('op'),
                'pixeltable.udf.tokens',
                {**dims, 'pxt.input_tokens': 7, 'pxt.output_tokens': 3},
            )
        ]

    def test_udf_call(self, telemetry_recorder: RecordingSubscriber) -> None:
        fn = pxtf.string.upper
        dims = {'pxt.udf': 'upper', 'pxt.udf_path': 'pixeltable.functions.string.upper'}
        # a successful call: one call counted, one latency recorded
        with telemetry_schemas.udf_call(fn):
            pass
        # a raising call counts and records the same way
        with pytest.raises(ValueError, match='boom'), telemetry_schemas.udf_call(fn):
            raise ValueError('boom')
        assert telemetry_recorder.counter_adds == [(telemetry_schemas.udf_calls, 1, dims)] * 2
        assert [(h, attrs) for h, _, attrs in telemetry_recorder.histogram_records] == [
            (telemetry_schemas.udf_latency, dims)
        ] * 2
        assert all(value >= 0 for _, value, _ in telemetry_recorder.histogram_records)

    def test_metrics_inactive_noop(self) -> None:
        assert not telemetry.active()
        telemetry.counter('pixeltable.test.rows').add(1, table='t')
        telemetry.histogram('pixeltable.test.latency').record(1.0)

    def test_metrics_exception_isolation(self, telemetry_recorder: RecordingSubscriber) -> None:
        raising = RaisingSubscriber()
        SubscriberRegistry.get().subscribe(raising)
        try:
            telemetry.counter('pixeltable.test.rows').add(1)
            telemetry.histogram('pixeltable.test.latency').record(1.0)
        finally:
            SubscriberRegistry.get().unsubscribe(raising)
        assert len(telemetry_recorder.counter_adds) == 1
        assert len(telemetry_recorder.histogram_records) == 1

    def test_ambient_nesting(self, telemetry_recorder: RecordingSubscriber) -> None:
        op = telemetry.span_start('op', set_current=True)
        assert telemetry.current_span() is op
        child = telemetry.span_start('child')
        telemetry.span_end(child)
        telemetry.span_end(op, attrs={'pxt.rows': 3})
        assert telemetry.current_span() is None

        assert telemetry_recorder.find('child')['parent_id'] == telemetry_recorder.find('op')['id']
        assert telemetry_recorder.find('op')['end_attrs'] == {'pxt.rows': 3}
        assert all(s['ended'] for s in telemetry_recorder.spans)

    def test_root_suppression(self, telemetry_recorder: RecordingSubscriber) -> None:
        # only operation spans (set_current=True) may be roots
        orphan = telemetry.span_start('orphan')
        assert not isinstance(orphan, telemetry.SpanHandle)
        child = telemetry.span_start('child', parent=orphan)
        telemetry.span_end(child)
        telemetry.span_end(orphan)
        assert telemetry_recorder.spans == []

    def test_explicit_parent(self, telemetry_recorder: RecordingSubscriber) -> None:
        op = telemetry.span_start('op', set_current=True)
        node = telemetry.span_start('node')
        # explicit parent overrides the ambient op span
        udf = telemetry.span_start('udf', parent=node)
        telemetry.span_end(udf)
        telemetry.span_end(node)
        telemetry.span_end(op)
        assert telemetry_recorder.find('node')['parent_id'] == telemetry_recorder.find('op')['id']
        assert telemetry_recorder.find('udf')['parent_id'] == telemetry_recorder.find('node')['id']

    def test_level_threshold_passthrough(self, telemetry_recorder: RecordingSubscriber) -> None:
        op = telemetry.span_start('op', set_current=True)
        suppressed = telemetry.span_start('batch', level=telemetry.DEBUG)
        assert not isinstance(suppressed, telemetry.SpanHandle)
        # a suppressed span returns None; its children fall back to the ambient span
        child = telemetry.span_start('child', parent=suppressed)
        telemetry.span_end(child)
        telemetry.span_end(suppressed)  # no-op
        telemetry.span_end(op)
        assert [s['name'] for s in telemetry_recorder.spans] == ['op', 'child']
        assert telemetry_recorder.find('child')['parent_id'] == telemetry_recorder.find('op')['id']

        telemetry.set_span_level(telemetry.TRACE)
        op2 = telemetry.span_start('op2', set_current=True)
        revealed = telemetry.span_start('batch', level=telemetry.DEBUG)
        assert isinstance(revealed, telemetry.SpanHandle)
        telemetry.span_end(revealed)
        telemetry.span_end(op2)
        assert telemetry_recorder.find('batch')['ended']

    def test_lazy_attrs(self, telemetry_recorder: RecordingSubscriber) -> None:
        calls: list[str] = []

        def attrs() -> dict[str, Any]:
            calls.append('attrs')
            return {'pxt.x': 1}

        suppressed = telemetry.span_start('a', level=telemetry.DEBUG, attrs=attrs)
        assert calls == []  # not resolved for suppressed spans
        telemetry.span_end(suppressed, attrs=attrs)
        assert calls == []
        emitted = telemetry.span_start('b', set_current=True, attrs=attrs)
        telemetry.span_end(emitted, attrs=attrs)
        assert calls == ['attrs', 'attrs']
        assert telemetry_recorder.find('b')['attrs'] == {'pxt.x': 1}

    def test_exception_isolation(self, telemetry_recorder: RecordingSubscriber) -> None:
        bad = RaisingSubscriber()
        SubscriberRegistry.get().subscribe(bad)
        try:
            handle = telemetry.span_start('a', set_current=True)
            telemetry.emit(handle, 'e', k=1)
            telemetry.span_end(handle)
            token = telemetry.restore_context(telemetry.capture_context())
            telemetry.exit_context(token)
        finally:
            SubscriberRegistry.get().unsubscribe(bad)
        # the well-behaved subscriber saw everything despite the raising one
        assert telemetry_recorder.find('a')['ended']
        assert telemetry_recorder.events == [(telemetry_recorder.find('a'), 'e', {'pxt.k': 1})]

    def test_late_subscriber(self, telemetry_recorder: RecordingSubscriber) -> None:
        op = telemetry.span_start('op', set_current=True)
        late = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(late)
        try:
            child = telemetry.span_start('child')
            telemetry.span_end(child)
            telemetry.span_end(op)
        finally:
            SubscriberRegistry.get().unsubscribe(late)
        # the late subscriber never sees an end without a start; its child span is a root
        assert [s['name'] for s in late.spans] == ['child']
        assert late.find('child')['parent_id'] is None
        assert late.find('child')['ended']
        assert telemetry_recorder.find('op')['ended']

    def test_equal_subscribers_use_identity_for_parent_tokens(self) -> None:
        class EqualSubscriber(RecordingSubscriber):
            def __eq__(self, other: object) -> bool:
                return isinstance(other, EqualSubscriber)

            def __hash__(self) -> int:
                return 0

        first = EqualSubscriber()
        second = EqualSubscriber()
        SubscriberRegistry.get().subscribe(first)
        try:
            op = telemetry.span_start('op', set_current=True)
            SubscriberRegistry.get().subscribe(second)
            try:
                child = telemetry.span_start('child')
                telemetry.span_end(child)
                telemetry.span_end(op)
            finally:
                SubscriberRegistry.get().unsubscribe(second)
        finally:
            SubscriberRegistry.get().unsubscribe(first)
        assert second.find('child')['parent_id'] is None

    def test_span_cm(self, telemetry_recorder: RecordingSubscriber) -> None:
        with telemetry.span('work', set_current=True, rows=2, skipped=None):
            pass
        record = telemetry_recorder.find('work')
        assert record['attrs'] == {'pxt.rows': 2}  # 'pxt.' prefix added, None skipped
        assert record['ended'] and record['exc'] is None

        with pytest.raises(ValueError, match='boom'), telemetry.span('failing', set_current=True):
            raise ValueError('boom')
        assert isinstance(telemetry_recorder.find('failing')['exc'], ValueError)

    def test_add_attrs(self, telemetry_recorder: RecordingSubscriber) -> None:
        with telemetry.span('op', set_current=True) as sp:
            telemetry.add_attrs(sp, rows=5, skipped=None)
        assert telemetry_recorder.find('op')['end_attrs'] == {'pxt.rows': 5}
        telemetry.add_attrs(None, rows=1)  # no-op for inactive/suppressed handles
        suppressed = telemetry.span_start('s', level=telemetry.DEBUG)
        telemetry.add_attrs(suppressed, rows=1)
        telemetry.span_end(suppressed)
        assert [s['name'] for s in telemetry_recorder.spans] == ['op']

    def test_emit(self, telemetry_recorder: RecordingSubscriber) -> None:
        handle = telemetry.span_start('op', set_current=True)
        telemetry.emit(handle, 'e', k=1, skipped=None)
        telemetry.span_end(handle)
        assert telemetry_recorder.events == [
            (telemetry_recorder.find('op'), 'e', {'pxt.k': 1})
        ]  # 'pxt.' prefix added, None skipped
        telemetry.emit(None, 'e', k=1)  # no-op for inactive/suppressed handles
        suppressed = telemetry.span_start('s', level=telemetry.DEBUG)
        telemetry.emit(suppressed, 'e', k=1)
        assert len(telemetry_recorder.events) == 1

    def test_spanned(self, telemetry_recorder: RecordingSubscriber) -> None:
        @telemetry.spanned('work', set_current=True)
        def fn(x: int) -> int:
            return x + 1

        assert fn(1) == 2
        record = telemetry_recorder.find('work')
        assert record['set_current'] is True and record['ended'] and record['exc'] is None

        @telemetry.spanned('failing', set_current=True)
        def boom() -> None:
            raise ValueError('boom')

        with pytest.raises(ValueError, match='boom'):
            boom()
        assert isinstance(telemetry_recorder.find('failing')['exc'], ValueError)

    def test_spanned_preserves_topology(self, telemetry_recorder: RecordingSubscriber) -> None:
        # a spanned(set_current=False) function must not become the ambient parent: spans started inside it
        # parent to the ambient op span, exactly as with a lexical `with span(...)` block
        @telemetry.spanned('node')
        def fn() -> None:
            inner = telemetry.span_start('inner')
            telemetry.span_end(inner)

        op = telemetry.span_start('op', set_current=True)
        fn()
        telemetry.span_end(op)
        assert telemetry_recorder.find('node')['parent_id'] == telemetry_recorder.find('op')['id']
        assert telemetry_recorder.find('inner')['parent_id'] == telemetry_recorder.find('op')['id']

        @telemetry.spanned('nesting', nest_children=True)
        def nesting_fn() -> None:
            inner = telemetry.span_start('nested_inner')
            telemetry.span_end(inner)

        # nest_children=True inside an op span: spans started inside the function nest under it
        op = telemetry.span_start('op2', set_current=True)
        nesting_fn()
        telemetry.span_end(op)
        assert telemetry_recorder.find('nesting')['parent_id'] == telemetry_recorder.find('op2')['id']
        assert telemetry_recorder.find('nested_inner')['parent_id'] == telemetry_recorder.find('nesting')['id']
        # nest_children=True outside any op span: the function's span is suppressed rather than becoming a root
        num_spans = len(telemetry_recorder.spans)
        nesting_fn()
        assert len(telemetry_recorder.spans) == num_spans

    def test_spanned_func_span(self, telemetry_recorder: RecordingSubscriber) -> None:
        assert telemetry.func_span() is None

        @telemetry.spanned('work', set_current=True)
        def fn() -> None:
            telemetry.add_attrs(telemetry.func_span(), rows=5)

        fn()
        assert telemetry_recorder.find('work')['end_attrs'] == {'pxt.rows': 5}
        assert telemetry.func_span() is None

        # a suppressed spanned() span shadows any outer one, so attrs can't leak to the wrong span
        @telemetry.spanned('suppressed', level=telemetry.DEBUG)
        def inner() -> None:
            assert telemetry.func_span() is None
            telemetry.add_attrs(telemetry.func_span(), rows=1)

        @telemetry.spanned('outer', set_current=True)
        def outer() -> None:
            inner()

        outer()
        assert telemetry_recorder.find('outer')['end_attrs'] is None

    def test_spanned_inactive(self) -> None:
        assert not telemetry.active()

        @telemetry.spanned('work', set_current=True)
        def fn(x: int) -> int:
            assert telemetry.func_span() is None
            return x * 2

        assert fn(3) == 6

    def test_failing_attrs_callable(self, telemetry_recorder: RecordingSubscriber) -> None:
        def bad() -> dict[str, Any]:
            raise RuntimeError('bad attrs')

        handle = telemetry.span_start('a', set_current=True, attrs=bad)
        telemetry.span_end(handle, attrs=bad)
        record = telemetry_recorder.find('a')
        assert record['attrs'] is None and record['ended']

    def test_thread_handoff(self, telemetry_recorder: RecordingSubscriber) -> None:
        op = telemetry.span_start('op', set_current=True)
        snapshot = telemetry.capture_context()

        def run() -> None:
            token = telemetry.restore_context(snapshot)
            try:
                child = telemetry.span_start('child')
                telemetry.span_end(child)
            finally:
                telemetry.exit_context(token)
            assert telemetry.current_span() is None

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        telemetry.span_end(op)
        assert telemetry_recorder.find('child')['parent_id'] == telemetry_recorder.find('op')['id']
        assert telemetry_recorder.ctx_restores == ['ctx']
        assert telemetry_recorder.ctx_exits == ['token']

    def test_concurrent_subscribe_unsubscribe(self, telemetry_recorder: RecordingSubscriber) -> None:
        # spans snapshot the registry at span_start; late subscribe/unsubscribe must not corrupt span_end
        # (SpanHandle pairs subs with tokens via zip(strict=True))
        stop = threading.Event()
        errors: list[Exception] = []

        def churn() -> None:
            try:
                while not stop.is_set():
                    s = RecordingSubscriber()
                    SubscriberRegistry.get().subscribe(s)
                    SubscriberRegistry.get().unsubscribe(s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=churn) for _ in range(2)]
        for t in threads:
            t.start()
        try:
            for _ in range(1000):
                op = telemetry.span_start('op', set_current=True)
                child = telemetry.span_start('child')
                telemetry.span_end(child)
                telemetry.span_end(op)
        finally:
            stop.set()
            for t in threads:
                t.join()
        assert errors == []
        assert all(s['ended'] for s in telemetry_recorder.spans)

    def test_asyncio_task_propagation(self, telemetry_recorder: RecordingSubscriber) -> None:
        async def main() -> None:
            op = telemetry.span_start('op', set_current=True)

            async def task() -> None:
                await asyncio.sleep(0)
                child = telemetry.span_start('child')
                telemetry.span_end(child)

            await asyncio.create_task(task())
            telemetry.span_end(op)

        asyncio.run(main())
        assert telemetry_recorder.find('child')['parent_id'] == telemetry_recorder.find('op')['id']
