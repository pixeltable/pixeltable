import pixeltable as pxt
from pixeltable import hooks

from .test_hooks import RecordingSubscriber


@pxt.udf
def add_one(x: int) -> int:
    return x + 1


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
            udfs = [s for s in sub.spans if s['name'] == 'udf.add_one']
            assert len(udfs) == 5
            assert all(u['parent_id'] in row_ids for u in udfs)
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
            assert [s for s in sub.spans if s['name'].startswith('udf.')] == []
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
            assert [s for s in sub.spans if s['name'].startswith('udf.')] == []
        finally:
            hooks.unsubscribe(sub)
            hooks.set_span_level(hooks.INFO)
