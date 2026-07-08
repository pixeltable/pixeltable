import asyncio
import threading
from typing import Any, Iterator

import pytest

from pixeltable import hooks


class RecordingSubscriber(hooks.Subscriber):
    """Records every hook invocation; span tokens are dict records linked by id."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self.events: list[tuple[str, dict[str, Any] | None]] = []
        self.counter_adds: list[tuple[hooks.Counter, int | float, dict[str, Any]]] = []
        self.histogram_records: list[tuple[hooks.Histogram, int | float, dict[str, Any]]] = []
        self.ctx_restores: list[Any] = []
        self.ctx_exits: list[Any] = []

    def on_span_start(self, name: str, parent_token: Any, attrs: dict[str, Any] | None, set_current: bool) -> Any:
        record: dict[str, Any] = {
            'id': len(self.spans),
            'name': name,
            'parent_id': parent_token['id'] if parent_token is not None else None,
            'attrs': attrs,
            'set_current': set_current,
            'ended': False,
            'exc': None,
            'end_attrs': None,
        }
        self.spans.append(record)
        return record

    def on_span_end(self, token: Any, exc: BaseException | None, attrs: dict[str, Any] | None) -> None:
        token['ended'] = True
        token['exc'] = exc
        token['end_attrs'] = attrs

    def on_event(self, name: str, attrs: dict[str, Any] | None) -> None:
        self.events.append((name, attrs))

    def on_counter_add(self, counter: hooks.Counter, value: int | float, attrs: dict[str, Any]) -> None:
        self.counter_adds.append((counter, value, attrs))

    def on_histogram_record(self, histogram: hooks.Histogram, value: int | float, attrs: dict[str, Any]) -> None:
        self.histogram_records.append((histogram, value, attrs))

    def capture_context(self) -> Any:
        return 'ctx'

    def restore_context(self, ctx: Any) -> Any:
        self.ctx_restores.append(ctx)
        return 'token'

    def exit_context(self, token: Any) -> None:
        self.ctx_exits.append(token)

    def find(self, name: str) -> dict[str, Any]:
        matches = [s for s in self.spans if s['name'] == name]
        assert len(matches) == 1, f'expected exactly one span named {name!r}, got {len(matches)}'
        return matches[0]


class RaisingSubscriber(hooks.Subscriber):
    def on_span_start(self, name: str, parent_token: Any, attrs: dict[str, Any] | None, set_current: bool) -> Any:
        raise RuntimeError('on_span_start')

    def on_span_end(self, token: Any, exc: BaseException | None, attrs: dict[str, Any] | None) -> None:
        raise RuntimeError('on_span_end')

    def on_event(self, name: str, attrs: dict[str, Any] | None) -> None:
        raise RuntimeError('on_event')

    def on_counter_add(self, counter: hooks.Counter, value: int | float, attrs: dict[str, Any]) -> None:
        raise RuntimeError('on_counter_add')

    def on_histogram_record(self, histogram: hooks.Histogram, value: int | float, attrs: dict[str, Any]) -> None:
        raise RuntimeError('on_histogram_record')

    def capture_context(self) -> Any:
        raise RuntimeError('capture_context')

    def restore_context(self, ctx: Any) -> Any:
        raise RuntimeError('restore_context')

    def exit_context(self, token: Any) -> None:
        raise RuntimeError('exit_context')


@pytest.fixture
def sub() -> Iterator[RecordingSubscriber]:
    s = RecordingSubscriber()
    hooks.subscribe(s)
    yield s
    hooks.unsubscribe(s)
    hooks.set_span_level(hooks.INFO)
    # a test failing between span_start(set_current=True) and span_end would leak the ambient span into
    # subsequent tests
    hooks._current_span.set(None)


class TestHooks:
    def test_inactive_is_noop(self) -> None:
        assert not hooks.active()
        calls: list[str] = []

        def attrs() -> dict[str, Any]:
            calls.append('attrs')
            return {}

        handle = hooks.span_start('a', attrs=attrs)
        assert handle is None
        hooks.span_end(handle)
        hooks.emit('e', attrs=attrs)
        assert hooks.capture_context() is None
        hooks.restore_context(None)
        hooks.exit_context(None)
        assert calls == []

    def test_subscribe_unsubscribe(self) -> None:
        s = RecordingSubscriber()
        hooks.subscribe(s)
        hooks.subscribe(s)  # idempotent
        try:
            assert hooks.active()
            handle = hooks.span_start('a', set_current=True)
            hooks.span_end(handle)
            assert len(s.spans) == 1
        finally:
            hooks.unsubscribe(s)
        assert not hooks.active()
        assert hooks.span_start('b', set_current=True) is None
        assert len(s.spans) == 1

    def test_counter_and_histogram(self, sub: RecordingSubscriber) -> None:
        c = hooks.counter('pixeltable.test.rows', unit='{row}')
        h = hooks.histogram('pixeltable.test.latency', unit='s')
        c.add(5, table='dir.tbl', skipped=None)
        h.record(0.25, udf='f')
        assert sub.counter_adds == [(c, 5, {'pxt.table': 'dir.tbl'})]
        assert sub.histogram_records == [(h, 0.25, {'pxt.udf': 'f'})]

    def test_metrics_inactive_noop(self) -> None:
        assert not hooks.active()
        hooks.counter('pixeltable.test.rows').add(1, table='t')
        hooks.histogram('pixeltable.test.latency').record(1.0)

    def test_metrics_exception_isolation(self, sub: RecordingSubscriber) -> None:
        raising = RaisingSubscriber()
        hooks.subscribe(raising)
        try:
            hooks.counter('pixeltable.test.rows').add(1)
            hooks.histogram('pixeltable.test.latency').record(1.0)
        finally:
            hooks.unsubscribe(raising)
        assert len(sub.counter_adds) == 1
        assert len(sub.histogram_records) == 1

    def test_ambient_nesting(self, sub: RecordingSubscriber) -> None:
        op = hooks.span_start('op', set_current=True)
        assert hooks.current_span() is op
        child = hooks.span_start('child')
        hooks.span_end(child)
        hooks.span_end(op, attrs={'pxt.rows': 3})
        assert hooks.current_span() is None

        assert sub.find('child')['parent_id'] == sub.find('op')['id']
        assert sub.find('op')['end_attrs'] == {'pxt.rows': 3}
        assert all(s['ended'] for s in sub.spans)

    def test_root_suppression(self, sub: RecordingSubscriber) -> None:
        # only operation spans (set_current=True) may be roots
        orphan = hooks.span_start('orphan')
        assert not isinstance(orphan, hooks.SpanHandle)
        child = hooks.span_start('child', parent=orphan)
        hooks.span_end(child)
        hooks.span_end(orphan)
        assert sub.spans == []

    def test_explicit_parent(self, sub: RecordingSubscriber) -> None:
        op = hooks.span_start('op', set_current=True)
        node = hooks.span_start('node')
        # explicit parent overrides the ambient op span
        udf = hooks.span_start('udf', parent=node)
        hooks.span_end(udf)
        hooks.span_end(node)
        hooks.span_end(op)
        assert sub.find('node')['parent_id'] == sub.find('op')['id']
        assert sub.find('udf')['parent_id'] == sub.find('node')['id']

    def test_level_threshold_passthrough(self, sub: RecordingSubscriber) -> None:
        op = hooks.span_start('op', set_current=True)
        suppressed = hooks.span_start('batch', level=hooks.DEBUG)
        assert not isinstance(suppressed, hooks.SpanHandle)
        # a suppressed span returns None; its children fall back to the ambient span
        child = hooks.span_start('child', parent=suppressed)
        hooks.span_end(child)
        hooks.span_end(suppressed)  # no-op
        hooks.span_end(op)
        assert [s['name'] for s in sub.spans] == ['op', 'child']
        assert sub.find('child')['parent_id'] == sub.find('op')['id']

        hooks.set_span_level(hooks.TRACE)
        op2 = hooks.span_start('op2', set_current=True)
        revealed = hooks.span_start('batch', level=hooks.DEBUG)
        assert isinstance(revealed, hooks.SpanHandle)
        hooks.span_end(revealed)
        hooks.span_end(op2)
        assert sub.find('batch')['ended']

    def test_lazy_attrs(self, sub: RecordingSubscriber) -> None:
        calls: list[str] = []

        def attrs() -> dict[str, Any]:
            calls.append('attrs')
            return {'pxt.x': 1}

        suppressed = hooks.span_start('a', level=hooks.DEBUG, attrs=attrs)
        assert calls == []  # not resolved for suppressed spans
        hooks.span_end(suppressed, attrs=attrs)
        assert calls == []
        emitted = hooks.span_start('b', set_current=True, attrs=attrs)
        hooks.span_end(emitted, attrs=attrs)
        assert calls == ['attrs', 'attrs']
        assert sub.find('b')['attrs'] == {'pxt.x': 1}

    def test_exception_isolation(self, sub: RecordingSubscriber) -> None:
        bad = RaisingSubscriber()
        hooks.subscribe(bad)
        try:
            handle = hooks.span_start('a', set_current=True)
            hooks.span_end(handle)
            hooks.emit('e', attrs={'k': 1})
            token = hooks.restore_context(hooks.capture_context())
            hooks.exit_context(token)
        finally:
            hooks.unsubscribe(bad)
        # the well-behaved subscriber saw everything despite the raising one
        assert sub.find('a')['ended']
        assert sub.events == [('e', {'k': 1})]

    def test_late_subscriber(self, sub: RecordingSubscriber) -> None:
        op = hooks.span_start('op', set_current=True)
        late = RecordingSubscriber()
        hooks.subscribe(late)
        try:
            child = hooks.span_start('child')
            hooks.span_end(child)
            hooks.span_end(op)
        finally:
            hooks.unsubscribe(late)
        # the late subscriber never sees an end without a start; its child span is a root
        assert [s['name'] for s in late.spans] == ['child']
        assert late.find('child')['parent_id'] is None
        assert late.find('child')['ended']
        assert sub.find('op')['ended']

    def test_span_cm(self, sub: RecordingSubscriber) -> None:
        with hooks.span('work', set_current=True, rows=2, skipped=None):
            pass
        record = sub.find('work')
        assert record['attrs'] == {'pxt.rows': 2}  # 'pxt.' prefix added, None skipped
        assert record['ended'] and record['exc'] is None

        with pytest.raises(ValueError, match='boom'), hooks.span('failing', set_current=True):
            raise ValueError('boom')
        assert isinstance(sub.find('failing')['exc'], ValueError)

    def test_add_attrs(self, sub: RecordingSubscriber) -> None:
        with hooks.span('op', set_current=True) as sp:
            hooks.add_attrs(sp, rows=5, skipped=None)
        assert sub.find('op')['end_attrs'] == {'pxt.rows': 5}
        hooks.add_attrs(None, rows=1)  # no-op for inactive/suppressed handles
        suppressed = hooks.span_start('s', level=hooks.DEBUG)
        hooks.add_attrs(suppressed, rows=1)
        hooks.span_end(suppressed)
        assert [s['name'] for s in sub.spans] == ['op']

    def test_spanned(self, sub: RecordingSubscriber) -> None:
        @hooks.spanned('work', set_current=True)
        def fn(x: int) -> int:
            return x + 1

        assert fn(1) == 2
        record = sub.find('work')
        assert record['set_current'] is True and record['ended'] and record['exc'] is None

        @hooks.spanned('failing', set_current=True)
        def boom() -> None:
            raise ValueError('boom')

        with pytest.raises(ValueError, match='boom'):
            boom()
        assert isinstance(sub.find('failing')['exc'], ValueError)

    def test_spanned_preserves_topology(self, sub: RecordingSubscriber) -> None:
        # a spanned(set_current=False) function must not become the ambient parent: spans started inside it
        # parent to the ambient op span, exactly as with a lexical `with span(...)` block
        @hooks.spanned('node')
        def fn() -> None:
            inner = hooks.span_start('inner')
            hooks.span_end(inner)

        op = hooks.span_start('op', set_current=True)
        fn()
        hooks.span_end(op)
        assert sub.find('node')['parent_id'] == sub.find('op')['id']
        assert sub.find('inner')['parent_id'] == sub.find('op')['id']

    def test_spanned_func_span(self, sub: RecordingSubscriber) -> None:
        assert hooks.func_span() is None

        @hooks.spanned('work', set_current=True)
        def fn() -> None:
            hooks.add_attrs(hooks.func_span(), rows=5)

        fn()
        assert sub.find('work')['end_attrs'] == {'pxt.rows': 5}
        assert hooks.func_span() is None

        # a suppressed spanned() span shadows any outer one, so attrs can't leak to the wrong span
        @hooks.spanned('suppressed', level=hooks.DEBUG)
        def inner() -> None:
            assert hooks.func_span() is None
            hooks.add_attrs(hooks.func_span(), rows=1)

        @hooks.spanned('outer', set_current=True)
        def outer() -> None:
            inner()

        outer()
        assert sub.find('outer')['end_attrs'] is None

    def test_spanned_inactive(self) -> None:
        assert not hooks.active()

        @hooks.spanned('work', set_current=True)
        def fn(x: int) -> int:
            assert hooks.func_span() is None
            return x * 2

        assert fn(3) == 6

    def test_failing_attrs_callable(self, sub: RecordingSubscriber) -> None:
        def bad() -> dict[str, Any]:
            raise RuntimeError('bad attrs')

        handle = hooks.span_start('a', set_current=True, attrs=bad)
        hooks.span_end(handle, attrs=bad)
        record = sub.find('a')
        assert record['attrs'] is None and record['ended']
        hooks.emit('e', attrs=bad)
        assert sub.events == [('e', None)]

    def test_thread_handoff(self, sub: RecordingSubscriber) -> None:
        op = hooks.span_start('op', set_current=True)
        snapshot = hooks.capture_context()

        def run() -> None:
            token = hooks.restore_context(snapshot)
            try:
                child = hooks.span_start('child')
                hooks.span_end(child)
            finally:
                hooks.exit_context(token)
            assert hooks.current_span() is None

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        hooks.span_end(op)
        assert sub.find('child')['parent_id'] == sub.find('op')['id']
        assert sub.ctx_restores == ['ctx']
        assert sub.ctx_exits == ['token']

    def test_concurrent_subscribe_unsubscribe(self, sub: RecordingSubscriber) -> None:
        # spans snapshot the registry at span_start; late subscribe/unsubscribe must not corrupt span_end
        # (SpanHandle pairs subs with tokens via zip(strict=True))
        stop = threading.Event()
        errors: list[Exception] = []

        def churn() -> None:
            try:
                while not stop.is_set():
                    s = RecordingSubscriber()
                    hooks.subscribe(s)
                    hooks.unsubscribe(s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=churn) for _ in range(2)]
        for t in threads:
            t.start()
        try:
            for _ in range(1000):
                op = hooks.span_start('op', set_current=True)
                child = hooks.span_start('child')
                hooks.span_end(child)
                hooks.span_end(op)
        finally:
            stop.set()
            for t in threads:
                t.join()
        assert errors == []
        assert all(s['ended'] for s in sub.spans)

    def test_asyncio_task_propagation(self, sub: RecordingSubscriber) -> None:
        async def main() -> None:
            op = hooks.span_start('op', set_current=True)

            async def task() -> None:
                await asyncio.sleep(0)
                child = hooks.span_start('child')
                hooks.span_end(child)

            await asyncio.create_task(task())
            hooks.span_end(op)

        asyncio.run(main())
        assert sub.find('child')['parent_id'] == sub.find('op')['id']
