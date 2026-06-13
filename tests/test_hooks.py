import asyncio
import threading
from typing import Any, Iterator

import pytest

import pixeltable as pxt
from pixeltable import hooks


@pxt.udf
def _double(x: int) -> int:
    return x * 2


@pxt.udf
def _fail_on_neg(x: int) -> int:
    if x < 0:
        raise ValueError('negative input')
    return x


class RecordingSubscriber(hooks.Subscriber):
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


class RaisingSubscriber(hooks.Subscriber):
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
    hooks.subscribe(s)
    yield s
    hooks.unsubscribe(s)
    hooks.set_span_level(hooks.INFO)


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
        # children of a suppressed span parent to the nearest emitted ancestor
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


class TestHooksIntegration:
    """End-to-end checks that table operations produce the expected spans/events."""

    def test_insert_span_tree(self, uses_db: None, sub: RecordingSubscriber) -> None:
        t = pxt.create_table('hooks_test', {'a': pxt.Int})
        t.add_computed_column(b=_double(t.a))
        sub.spans.clear()
        sub.events.clear()

        status = t.insert([{'a': i} for i in range(5)])
        assert status.num_rows == 5

        insert_span = sub.find('pixeltable.insert')
        assert insert_span['ended'] and insert_span['exc'] is None
        assert insert_span['end_attrs']['pxt.table'] == 'hooks_test'
        assert insert_span['end_attrs']['pxt.rows'] == 5

        # begin_xact, store spans, and the plan head are children of the insert span; the rest of the
        # exec chain nests under its consumer node
        names_under_insert = {s['name'] for s in sub.spans if s['parent_id'] == insert_span['id']}
        assert 'catalog.begin_xact' in names_under_insert
        assert 'exec.ExprEvalNode' in names_under_insert
        assert 'store.sql_insert' in names_under_insert
        eval_span = sub.find('exec.ExprEvalNode')
        assert sub.find('exec.InMemoryDataNode')['parent_id'] == eval_span['id']
        assert all(s['ended'] for s in sub.spans)

        # the ExprEvalNode span carries aggregated per-udf stats, keyed by column name
        end_attrs = eval_span['end_attrs']
        assert end_attrs['pxt.udf.b.count'] == 5
        assert end_attrs['pxt.udf.b.total_s'] >= 0
        assert end_attrs['pxt.udf.b.avg_s'] >= 0
        assert 'pxt.udf.b.fn' in end_attrs
        # at the default INFO level there are no per-call udf spans
        assert not any(s['name'].startswith('udf.') for s in sub.spans)

        event_names = [name for name, _ in sub.events]
        assert 'rows.written' in event_names
        assert 'cells.computed' in event_names
        udf_stats = [attrs for name, attrs in sub.events if name == 'udf.stats']
        assert len(udf_stats) == 1
        assert udf_stats[0]['pxt.column'] == 'b'
        assert udf_stats[0]['count'] == 5

    def test_cell_errors(self, uses_db: None, sub: RecordingSubscriber) -> None:
        t = pxt.create_table('hooks_test', {'a': pxt.Int})
        t.add_computed_column(b=_fail_on_neg(t.a))
        sub.spans.clear()
        sub.events.clear()

        status = t.insert([{'a': 1}, {'a': -1}, {'a': -2}], on_error='ignore')
        assert status.num_excs > 0

        cell_errors = [attrs for name, attrs in sub.events if name == 'cell.error']
        assert sum(e['count'] for e in cell_errors) == 2
        assert all(e['pxt.column'] == 'b' and e['error_type'] == 'ValueError' for e in cell_errors)
        eval_span = sub.find('exec.ExprEvalNode')
        assert eval_span['end_attrs']['pxt.udf.b.errors'] == 2
        assert sub.find('pixeltable.insert')['exc'] is None

    def test_insert_abort_records_exc(self, uses_db: None, sub: RecordingSubscriber) -> None:
        t = pxt.create_table('hooks_test', {'a': pxt.Int})
        t.add_computed_column(b=_fail_on_neg(t.a))
        sub.spans.clear()

        with pytest.raises(pxt.exceptions.ExprEvalError):
            t.insert([{'a': -1}], on_error='abort')
        insert_span = sub.find('pixeltable.insert')
        assert insert_span['ended'] and insert_span['exc'] is not None

    def test_trace_level_udf_spans(self, uses_db: None, sub: RecordingSubscriber) -> None:
        t = pxt.create_table('hooks_test', {'a': pxt.Int})
        t.add_computed_column(b=_double(t.a))
        sub.spans.clear()
        hooks.set_span_level(hooks.TRACE)

        t.insert([{'a': 1}, {'a': 2}])
        udf_spans = [s for s in sub.spans if s['name'].startswith('udf.')]
        assert len(udf_spans) == 2  # one per row at TRACE
        eval_span = sub.find('exec.ExprEvalNode')
        assert all(s['parent_id'] == eval_span['id'] for s in udf_spans)
        assert sub.find('store.build_rows')['ended']  # DEBUG span now visible
