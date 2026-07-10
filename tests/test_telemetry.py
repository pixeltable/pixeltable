import asyncio
import threading
from typing import Any, Iterator

import pytest

from pixeltable import telemetry
from pixeltable.telemetry import TelemetryEnv


class RecordingSubscriber(telemetry.Subscriber):
    """Records every hook invocation; span tokens are dict records linked by id."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self.events: list[tuple[str, dict[str, Any] | None]] = []
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


class RaisingSubscriber(telemetry.Subscriber):
    def on_span_start(self, name: str, parent_token: Any, attrs: dict[str, Any] | None, set_current: bool) -> Any:
        raise RuntimeError('on_span_start')

    def on_span_end(self, token: Any, exc: BaseException | None, attrs: dict[str, Any] | None) -> None:
        raise RuntimeError('on_span_end')

    def on_event(self, name: str, attrs: dict[str, Any] | None) -> None:
        raise RuntimeError('on_event')

    def capture_context(self) -> Any:
        raise RuntimeError('capture_context')

    def restore_context(self, ctx: Any) -> Any:
        raise RuntimeError('restore_context')

    def exit_context(self, token: Any) -> None:
        raise RuntimeError('exit_context')


@pytest.fixture
def sub() -> Iterator[RecordingSubscriber]:
    s = RecordingSubscriber()
    TelemetryEnv.get().subscribe(s)
    yield s
    TelemetryEnv.get().unsubscribe(s)
    telemetry.set_span_level(telemetry.INFO)
    # a test failing between span_start(set_current=True) and span_end would leak the ambient span into
    # subsequent tests
    telemetry._current_span.set(None)


class TestHooks:
    def test_inactive_is_noop(self) -> None:
        assert not TelemetryEnv.get().active()
        calls: list[str] = []

        def attrs() -> dict[str, Any]:
            calls.append('attrs')
            return {}

        handle = telemetry.span_start('a', attrs=attrs)
        assert handle is None
        telemetry.span_end(handle)
        telemetry.emit('e', attrs=attrs)
        assert telemetry.capture_context() is None
        telemetry.restore_context(None)
        telemetry.exit_context(None)
        assert calls == []

    def test_subscribe_unsubscribe(self) -> None:
        s = RecordingSubscriber()
        TelemetryEnv.get().subscribe(s)
        TelemetryEnv.get().subscribe(s)  # idempotent
        try:
            assert TelemetryEnv.get().active()
            handle = telemetry.span_start('a', set_current=True)
            telemetry.span_end(handle)
            assert len(s.spans) == 1
        finally:
            TelemetryEnv.get().unsubscribe(s)
        assert not TelemetryEnv.get().active()
        assert telemetry.span_start('b', set_current=True) is None
        assert len(s.spans) == 1

    def test_ambient_nesting(self, sub: RecordingSubscriber) -> None:
        op = telemetry.span_start('op', set_current=True)
        assert telemetry.current_span() is op
        child = telemetry.span_start('child')
        telemetry.span_end(child)
        telemetry.span_end(op, attrs={'pxt.rows': 3})
        assert telemetry.current_span() is None

        assert sub.find('child')['parent_id'] == sub.find('op')['id']
        assert sub.find('op')['end_attrs'] == {'pxt.rows': 3}
        assert all(s['ended'] for s in sub.spans)

    def test_root_suppression(self, sub: RecordingSubscriber) -> None:
        # only operation spans (set_current=True) may be roots
        orphan = telemetry.span_start('orphan')
        assert not isinstance(orphan, telemetry.SpanHandle)
        child = telemetry.span_start('child', parent=orphan)
        telemetry.span_end(child)
        telemetry.span_end(orphan)
        assert sub.spans == []

    def test_explicit_parent(self, sub: RecordingSubscriber) -> None:
        op = telemetry.span_start('op', set_current=True)
        node = telemetry.span_start('node')
        # explicit parent overrides the ambient op span
        udf = telemetry.span_start('udf', parent=node)
        telemetry.span_end(udf)
        telemetry.span_end(node)
        telemetry.span_end(op)
        assert sub.find('node')['parent_id'] == sub.find('op')['id']
        assert sub.find('udf')['parent_id'] == sub.find('node')['id']

    def test_level_threshold_passthrough(self, sub: RecordingSubscriber) -> None:
        op = telemetry.span_start('op', set_current=True)
        suppressed = telemetry.span_start('batch', level=telemetry.DEBUG)
        assert not isinstance(suppressed, telemetry.SpanHandle)
        # a suppressed span returns None; its children fall back to the ambient span
        child = telemetry.span_start('child', parent=suppressed)
        telemetry.span_end(child)
        telemetry.span_end(suppressed)  # no-op
        telemetry.span_end(op)
        assert [s['name'] for s in sub.spans] == ['op', 'child']
        assert sub.find('child')['parent_id'] == sub.find('op')['id']

        telemetry.set_span_level(telemetry.TRACE)
        op2 = telemetry.span_start('op2', set_current=True)
        revealed = telemetry.span_start('batch', level=telemetry.DEBUG)
        assert isinstance(revealed, telemetry.SpanHandle)
        telemetry.span_end(revealed)
        telemetry.span_end(op2)
        assert sub.find('batch')['ended']

    def test_lazy_attrs(self, sub: RecordingSubscriber) -> None:
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
        assert sub.find('b')['attrs'] == {'pxt.x': 1}

    def test_exception_isolation(self, sub: RecordingSubscriber) -> None:
        bad = RaisingSubscriber()
        TelemetryEnv.get().subscribe(bad)
        try:
            handle = telemetry.span_start('a', set_current=True)
            telemetry.span_end(handle)
            telemetry.emit('e', attrs={'k': 1})
            token = telemetry.restore_context(telemetry.capture_context())
            telemetry.exit_context(token)
        finally:
            TelemetryEnv.get().unsubscribe(bad)
        # the well-behaved subscriber saw everything despite the raising one
        assert sub.find('a')['ended']
        assert sub.events == [('e', {'k': 1})]

    def test_late_subscriber(self, sub: RecordingSubscriber) -> None:
        op = telemetry.span_start('op', set_current=True)
        late = RecordingSubscriber()
        TelemetryEnv.get().subscribe(late)
        try:
            child = telemetry.span_start('child')
            telemetry.span_end(child)
            telemetry.span_end(op)
        finally:
            TelemetryEnv.get().unsubscribe(late)
        # the late subscriber never sees an end without a start; its child span is a root
        assert [s['name'] for s in late.spans] == ['child']
        assert late.find('child')['parent_id'] is None
        assert late.find('child')['ended']
        assert sub.find('op')['ended']

    def test_span_cm(self, sub: RecordingSubscriber) -> None:
        with telemetry.span('work', set_current=True, rows=2, skipped=None):
            pass
        record = sub.find('work')
        assert record['attrs'] == {'pxt.rows': 2}  # 'pxt.' prefix added, None skipped
        assert record['ended'] and record['exc'] is None

        with pytest.raises(ValueError, match='boom'), telemetry.span('failing', set_current=True):
            raise ValueError('boom')
        assert isinstance(sub.find('failing')['exc'], ValueError)

    def test_add_attrs(self, sub: RecordingSubscriber) -> None:
        with telemetry.span('op', set_current=True) as sp:
            telemetry.add_attrs(sp, rows=5, skipped=None)
        assert sub.find('op')['end_attrs'] == {'pxt.rows': 5}
        telemetry.add_attrs(None, rows=1)  # no-op for inactive/suppressed handles
        suppressed = telemetry.span_start('s', level=telemetry.DEBUG)
        telemetry.add_attrs(suppressed, rows=1)
        telemetry.span_end(suppressed)
        assert [s['name'] for s in sub.spans] == ['op']

    def test_failing_attrs_callable(self, sub: RecordingSubscriber) -> None:
        def bad() -> dict[str, Any]:
            raise RuntimeError('bad attrs')

        handle = telemetry.span_start('a', set_current=True, attrs=bad)
        telemetry.span_end(handle, attrs=bad)
        record = sub.find('a')
        assert record['attrs'] is None and record['ended']
        telemetry.emit('e', attrs=bad)
        assert sub.events == [('e', None)]

    def test_thread_handoff(self, sub: RecordingSubscriber) -> None:
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
                    TelemetryEnv.get().subscribe(s)
                    TelemetryEnv.get().unsubscribe(s)
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
        assert all(s['ended'] for s in sub.spans)

    def test_asyncio_task_propagation(self, sub: RecordingSubscriber) -> None:
        async def main() -> None:
            op = telemetry.span_start('op', set_current=True)

            async def task() -> None:
                await asyncio.sleep(0)
                child = telemetry.span_start('child')
                telemetry.span_end(child)

            await asyncio.create_task(task())
            telemetry.span_end(op)

        asyncio.run(main())
        assert sub.find('child')['parent_id'] == sub.find('op')['id']
