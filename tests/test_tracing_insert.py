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
            rows = [s for s in sub.spans if s['name'] == 'pixeltable.row']
            assert len(rows) == 5
            assert all(r['parent_id'] == op['id'] for r in rows)
            row_ids = {r['id'] for r in rows}
            udfs = [s for s in sub.spans if s['name'] == 'pixeltable.udf.abs']
            assert len(udfs) == 5
            assert all(u['parent_id'] in row_ids for u in udfs)
            assert all(u['set_current'] for u in udfs)  # provider instrumentors must nest under the UDF span
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

    def test_bare_query_stays_dark(self, uses_db: None) -> None:
        """Shared machinery (row/udf spans) must not emit without a top-level operation span."""
        t = self._make_table()
        t.insert([{'c': i} for i in range(3)])

        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        telemetry.set_span_level(telemetry.DEBUG)
        try:
            # a query computes abs on the fly but has no operation span wrapping it
            _ = t.select(out=pxtf.math.abs(t.c)).collect()
            assert [s for s in sub.spans if s['name'] == 'pixeltable.row'] == []
            assert [s for s in sub.spans if s['name'].startswith('pixeltable.udf.')] == []
        finally:
            SubscriberRegistry.get().unsubscribe(sub)
            telemetry.set_span_level(telemetry.INFO)

    def test_non_insert_write_stays_dark(self, uses_db: None) -> None:
        """sql_insert is shared by uninstrumented write paths; those must not emit root spans."""
        t = self._make_table()
        t.insert([{'c': i} for i in range(3)])

        sub = RecordingSubscriber()
        SubscriberRegistry.get().subscribe(sub)
        try:
            t.update({'c': t.c + 1})
            assert [s for s in sub.spans if s['name'] == 'pixeltable.sa.insert_rows'] == []
        finally:
            SubscriberRegistry.get().unsubscribe(sub)
