import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import telemetry
from pixeltable.func import Batch
from pixeltable.telemetry import SubscriberRegistry

from .test_telemetry import RecordingSubscriber
from .utils import pxt_raises


@pxt.udf
def fail_on_three(x: int) -> int:
    if x == 3:
        raise ValueError('boom')
    return x + 1


@pxt.udf(batch_size=2)
def batch_add_one(vals: Batch[int]) -> Batch[int]:
    # We need this to validate that batched udfs have proper nesting
    with telemetry.span('provider.batch_add_one'):
        return [val + 1 for val in vals]


@pytest.mark.local('subscribes an in-process telemetry subscriber the daemon cannot see')
class TestInsertTracing:
    """End-to-end span nesting for the insert() path: operation -> row -> udf cell."""

    def _make_table(self) -> pxt.Table:
        t = pxt.create_table('tracing_test', {'c': pxt.Int}, if_exists='replace')
        t.add_computed_column(inc=pxtf.math.abs(t.c))
        return t

    def test_row_and_udf_spans_nest(self, uses_db: None) -> None:
        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        telemetry.set_span_level(telemetry.DEBUG)
        try:
            t = self._make_table()
            t.insert([{'c': i} for i in range(5)])

            op = sub.find('pixeltable.insert')
            assert op['set_current'] is True and op['parent_id'] is None
            assert op['end_attrs']['pxt.table_id'] == str(t._id)
            assert op['end_attrs']['pxt.num_rows'] == 5
            assert op['end_attrs']['pxt.num_excs'] == 0
            rows = [s for s in sub.spans if s['name'] == 'pixeltable.row']
            assert len(rows) == 5
            assert all(r['parent_id'] == op['id'] for r in rows)
            row_ids = {r['id'] for r in rows}
            udfs = [s for s in sub.spans if s['name'] == 'pixeltable.udf.abs']
            assert len(udfs) == 5
            assert all(u['parent_id'] in row_ids for u in udfs)
            assert all(u['set_current'] for u in udfs)  # provider instrumentors must nest under the UDF span
            assert all(u['attrs']['pxt.column'] == 'inc' for u in udfs)
            xacts = [
                s for s in sub.spans if s['name'] == 'pixeltable.catalog.begin_xact' and s['parent_id'] == op['id']
            ]
            assert any(
                s['end_attrs']['pxt.for_write'] and str(t._id) in s['end_attrs']['pxt.write_table_ids'] for s in xacts
            )
            assert all(s['end_attrs']['pxt.attempt'] == 0 for s in xacts)
            assert all(s['ended'] for s in sub.spans)
        finally:
            SubscriberRegistry.get().unsubscribe(sub)
            telemetry.set_span_level(telemetry.INFO)

    def test_debug_off_suppresses_row_and_udf_spans(self, uses_db: None) -> None:
        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        telemetry.set_span_level(telemetry.INFO)  # default: row/udf-cell spans are DEBUG, so suppressed
        try:
            t = self._make_table()
            t.insert([{'c': i} for i in range(5)])

            assert sub.find('pixeltable.insert')  # operation span (INFO) still emits
            assert [s for s in sub.spans if s['name'] == 'pixeltable.row'] == []
            assert [s for s in sub.spans if s['name'].startswith('pixeltable.udf.')] == []
        finally:
            SubscriberRegistry.get().unsubscribe(sub)

    def test_batched_udf_span_is_ambient(self, uses_db: None) -> None:
        t = pxt.create_table('tracing_test', {'c': pxt.Int}, if_exists='replace')
        t.add_computed_column(inc=batch_add_one(t.c))
        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        telemetry.set_span_level(telemetry.DEBUG)
        try:
            t.insert([{'c': i} for i in range(5)])
            udfs = [s for s in sub.spans if s['name'] == 'pixeltable.udf.batch_add_one']
            providers = [s for s in sub.spans if s['name'] == 'provider.batch_add_one']
            udf_ids = {s['id'] for s in udfs}
            assert len(udfs) == len(providers) == 3
            assert all(s['set_current'] for s in udfs)
            assert all(s['parent_id'] in udf_ids for s in providers)
            assert all(s['attrs']['pxt.column'] == 'inc' for s in udfs)
            assert sorted(s['attrs']['pxt.batch_size'] for s in udfs) == [1, 2, 2]  # 5 rows in batches of <= 2
        finally:
            SubscriberRegistry.get().unsubscribe(sub)
            telemetry.set_span_level(telemetry.INFO)

    def test_row_span_cap(self, uses_db: None) -> None:
        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        telemetry.set_span_level(telemetry.DEBUG)
        try:
            t = self._make_table()
            t.insert([{'c': i} for i in range(150)])

            rows = [s for s in sub.spans if s['name'] == 'pixeltable.row']
            assert 0 < len(rows) <= 100  # capped by MAX_ROW_SPANS, not one per input row
        finally:
            SubscriberRegistry.get().unsubscribe(sub)
            telemetry.set_span_level(telemetry.INFO)

    def test_failed_insert_records_exc(self, uses_db: None) -> None:
        t = pxt.create_table('tracing_test', {'c': pxt.Int}, if_exists='replace')
        t.add_computed_column(out=fail_on_three(t.c))
        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        try:
            with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='ValueError'):
                t.insert([{'c': i} for i in range(10)])
            op = sub.find('pixeltable.insert')
            assert op['ended']
            assert op['exc'] is not None
        finally:
            SubscriberRegistry.get().unsubscribe(sub)

    def test_failed_insert_ends_row_spans(self, uses_db: None) -> None:
        """Row spans opened before an abort must still be ended so subscribers see on_span_end()."""
        t = pxt.create_table('tracing_test', {'c': pxt.Int}, if_exists='replace')
        t.add_computed_column(out=fail_on_three(t.c))
        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        telemetry.set_span_level(telemetry.DEBUG)
        try:
            with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='ValueError'):
                t.insert([{'c': i} for i in range(10)])
            rows = [s for s in sub.spans if s['name'] == 'pixeltable.row']
            assert len(rows) > 0
            assert all(s['ended'] for s in sub.spans)
        finally:
            SubscriberRegistry.get().unsubscribe(sub)
            telemetry.set_span_level(telemetry.INFO)

    def test_collect_and_yield_rows_spans(self, uses_db: None) -> None:
        t = self._make_table()
        t.insert([{'c': i} for i in range(3)])

        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        telemetry.set_span_level(telemetry.DEBUG)
        try:
            _ = t.select(out=fail_on_three(t.c)).collect()

            collect = sub.find('pixeltable.collect')
            yield_rows = sub.find('pixeltable.result_cursor.yield_rows')
            assert collect['set_current'] is True and collect['parent_id'] is None
            assert collect['end_attrs']['pxt.tables'] == ['tracing_test']
            assert collect['end_attrs']['pxt.rows'] == 3
            assert yield_rows['set_current'] is True and yield_rows['parent_id'] == collect['id']
            assert yield_rows['attrs']['pxt.tables'] == ['tracing_test']
            assert yield_rows['end_attrs']['pxt.rows'] == 3
            rows = [s for s in sub.spans if s['name'] == 'pixeltable.row']
            assert len(rows) == 3
            assert all(row['parent_id'] == yield_rows['id'] for row in rows)
            row_ids = {row['id'] for row in rows}
            udfs = [s for s in sub.spans if s['name'] == 'pixeltable.udf.fail_on_three']
            assert len(udfs) == 3
            assert all(udf['parent_id'] in row_ids for udf in udfs)
            assert all(s['ended'] for s in sub.spans)
        finally:
            SubscriberRegistry.get().unsubscribe(sub)
            telemetry.set_span_level(telemetry.INFO)

    @pytest.mark.parametrize(('method', 'span_name'), [('head', 'pixeltable.head'), ('tail', 'pixeltable.tail')])
    def test_head_and_tail_spans(self, uses_db: None, method: str, span_name: str) -> None:
        t = self._make_table()
        t.insert([{'c': i} for i in range(3)])

        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        try:
            result = getattr(t, method)(2)
            assert len(result) == 2
            op = sub.find(span_name)
            yield_rows = sub.find('pixeltable.result_cursor.yield_rows')
            assert op['set_current'] is True and op['parent_id'] is None
            assert op['end_attrs']['pxt.tables'] == ['tracing_test']
            assert op['end_attrs']['pxt.rows'] == 2
            assert yield_rows['set_current'] is True and yield_rows['parent_id'] == op['id']
            assert all(s['ended'] for s in sub.spans)
        finally:
            SubscriberRegistry.get().unsubscribe(sub)

    def test_result_cursor_yield_rows_span(self, uses_db: None) -> None:
        t = self._make_table()
        t.insert([{'c': i} for i in range(3)])

        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        try:
            rows = list(t.select(t.c).cursor())
            assert len(rows) == 3
            yield_rows = sub.find('pixeltable.result_cursor.yield_rows')
            assert yield_rows['set_current'] is True and yield_rows['parent_id'] is None
            assert yield_rows['ended']
        finally:
            SubscriberRegistry.get().unsubscribe(sub)

    def test_count_and_move_spans(self, uses_db: None) -> None:
        t = self._make_table()
        t.insert([{'c': i} for i in range(3)])

        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        try:
            # count(): root op span carrying the counted rows
            assert t.where(t.c > 0).count() == 2
            op = sub.find('pixeltable.count')
            assert op['set_current'] is True and op['parent_id'] is None
            assert op['end_attrs']['pxt.tables'] == ['tracing_test']
            assert op['end_attrs']['pxt.rows'] == 2
            # move(): root op span carrying source and destination paths
            pxt.move('tracing_test', 'tracing_test2')
            op = sub.find('pixeltable.move')
            assert op['set_current'] is True and op['parent_id'] is None
            assert op['end_attrs']['pxt.path'] == 'tracing_test'
            assert op['end_attrs']['pxt.new_path'] == 'tracing_test2'
            assert all(s['ended'] for s in sub.spans)
        finally:
            SubscriberRegistry.get().unsubscribe(sub)

    def test_compute_span(self, uses_db: None) -> None:
        t = self._make_table()

        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        telemetry.set_span_level(telemetry.DEBUG)
        try:
            result = t.compute([{'c': i} for i in range(3)])
            assert len(result) == 3
            op = sub.find('pixeltable.compute')
            assert op['set_current'] is True and op['parent_id'] is None
            rows = [s for s in sub.spans if s['name'] == 'pixeltable.row']
            assert len(rows) == 3
            assert all(row['parent_id'] == op['id'] for row in rows)
            assert all(s['ended'] for s in sub.spans)
        finally:
            SubscriberRegistry.get().unsubscribe(sub)
            telemetry.set_span_level(telemetry.INFO)

    def test_update_emits_op_span(self, uses_db: None) -> None:
        """update() is an instrumented operation: work spans nest under its op span."""
        t = self._make_table()
        t.insert([{'c': i} for i in range(3)])

        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        try:
            t.update({'c': t.c + 1})
            op = sub.find('pixeltable.update')
            assert op['set_current'] is True and op['parent_id'] is None
            assert op['end_attrs']['pxt.table_id'] == str(t._id)
            assert op['end_attrs']['pxt.table'] == 'tracing_test'
            assert op['end_attrs']['pxt.version'] == 3  # v0 create, v1 add_computed_column, v2 insert, v3 update
            assert op['end_attrs']['pxt.num_rows'] == 3
            sa_spans = [s for s in sub.spans if s['name'] == 'pixeltable.sa.insert_rows']
            assert len(sa_spans) > 0
            assert all(s['parent_id'] == op['id'] for s in sa_spans)
            assert sum(s['attrs']['pxt.rows'] for s in sa_spans) == 3
        finally:
            SubscriberRegistry.get().unsubscribe(sub)
