import pytest

import pixeltable as pxt
from pixeltable import hooks

from .test_hooks import RecordingSubscriber
from .utils import pxt_raises


@pxt.udf
def add_one(x: int) -> int:
    return x + 1


@pxt.udf
def fail_on_three(x: int) -> int:
    if x == 3:
        raise ValueError('boom')
    return x + 1


@pytest.mark.local('subscribes an in-process hooks subscriber the daemon cannot see')
class TestInsertTracing:
    """End-to-end span nesting for the insert() path: operation -> row -> udf cell."""

    def _make_table(self) -> pxt.Table:
        t = pxt.create_table('tracing_test', {'c': pxt.Int}, if_exists='replace')
        t.add_computed_column(inc=add_one(t.c))
        return t

    def test_row_and_udf_spans_nest(self, uses_db: None) -> None:
        sub = RecordingSubscriber()
        hooks.subscribe(sub)
        hooks.set_span_level(hooks.DEBUG)
        try:
            t = self._make_table()
            t.insert([{'c': i} for i in range(5)])

            op = sub.find('pixeltable.insert')
            assert op['set_current'] is True and op['parent_id'] is None
            rows = [s for s in sub.spans if s['name'] == 'pixeltable.row']
            assert len(rows) == 5
            assert all(r['parent_id'] == op['id'] for r in rows)
            row_ids = {r['id'] for r in rows}
            udfs = [s for s in sub.spans if s['name'] == 'pixeltable.udf.add_one']
            assert len(udfs) == 5
            assert all(u['parent_id'] in row_ids for u in udfs)
            assert all(u['set_current'] for u in udfs)  # provider instrumentors must nest under the UDF span
            assert all(s['ended'] for s in sub.spans)
        finally:
            hooks.unsubscribe(sub)
            hooks.set_span_level(hooks.INFO)

    def test_debug_off_suppresses_row_and_udf_spans(self, uses_db: None) -> None:
        sub = RecordingSubscriber()
        hooks.subscribe(sub)
        hooks.set_span_level(hooks.INFO)  # default: row/udf-cell spans are DEBUG, so suppressed
        try:
            t = self._make_table()
            t.insert([{'c': i} for i in range(5)])

            assert sub.find('pixeltable.insert')  # operation span (INFO) still emits
            assert [s for s in sub.spans if s['name'] == 'pixeltable.row'] == []
            assert [s for s in sub.spans if s['name'].startswith('pixeltable.udf.')] == []
        finally:
            hooks.unsubscribe(sub)

    def test_row_span_cap(self, uses_db: None) -> None:
        sub = RecordingSubscriber()
        hooks.subscribe(sub)
        hooks.set_span_level(hooks.DEBUG)
        try:
            t = self._make_table()
            t.insert([{'c': i} for i in range(150)])

            rows = [s for s in sub.spans if s['name'] == 'pixeltable.row']
            assert 0 < len(rows) <= 100  # capped by MAX_ROW_SPANS, not one per input row
        finally:
            hooks.unsubscribe(sub)
            hooks.set_span_level(hooks.INFO)

    def test_failed_insert_records_exc(self, uses_db: None) -> None:
        t = pxt.create_table('tracing_test', {'c': pxt.Int}, if_exists='replace')
        t.add_computed_column(out=fail_on_three(t.c))
        sub = RecordingSubscriber()
        hooks.subscribe(sub)
        try:
            with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='ValueError'):
                t.insert([{'c': i} for i in range(10)])
            op = sub.find('pixeltable.insert')
            assert op['ended']
            assert op['exc'] is not None
        finally:
            hooks.unsubscribe(sub)

    def test_failed_insert_ends_row_spans(self, uses_db: None) -> None:
        """Row spans opened before an abort must still be ended so subscribers see on_span_end()."""
        t = pxt.create_table('tracing_test', {'c': pxt.Int}, if_exists='replace')
        t.add_computed_column(out=fail_on_three(t.c))
        sub = RecordingSubscriber()
        hooks.subscribe(sub)
        hooks.set_span_level(hooks.DEBUG)
        try:
            with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='ValueError'):
                t.insert([{'c': i} for i in range(10)])
            rows = [s for s in sub.spans if s['name'] == 'pixeltable.row']
            assert len(rows) > 0
            assert all(s['ended'] for s in sub.spans)
        finally:
            hooks.unsubscribe(sub)
            hooks.set_span_level(hooks.INFO)

    def test_bare_query_stays_dark(self, uses_db: None) -> None:
        """Shared machinery (row/udf spans) must not emit without a top-level operation span."""
        t = self._make_table()
        t.insert([{'c': i} for i in range(3)])

        sub = RecordingSubscriber()
        hooks.subscribe(sub)
        hooks.set_span_level(hooks.DEBUG)
        try:
            # a query computes add_one on the fly but has no operation span wrapping it
            _ = t.select(out=add_one(t.c)).collect()
            assert [s for s in sub.spans if s['name'] == 'pixeltable.row'] == []
            assert [s for s in sub.spans if s['name'].startswith('pixeltable.udf.')] == []
        finally:
            hooks.unsubscribe(sub)
            hooks.set_span_level(hooks.INFO)

    def test_update_emits_op_span(self, uses_db: None) -> None:
        """update() is an instrumented operation: work spans nest under its op span."""
        t = self._make_table()
        t.insert([{'c': i} for i in range(3)])

        sub = RecordingSubscriber()
        hooks.subscribe(sub)
        try:
            t.update({'c': t.c + 1})
            op = sub.find('pixeltable.update')
            assert op['set_current'] is True and op['parent_id'] is None
            sa_spans = [s for s in sub.spans if s['name'] == 'pixeltable.sa.insert_rows']
            assert len(sa_spans) > 0
            assert all(s['parent_id'] == op['id'] for s in sa_spans)
        finally:
            hooks.unsubscribe(sub)
