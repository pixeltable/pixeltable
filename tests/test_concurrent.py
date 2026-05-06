import threading
from typing import Callable

import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from tests.utils import DummyIterator, validate_update_status


def _run_workers(target: Callable[[int], None], n_threads: int) -> list[tuple[int, BaseException]]:
    """Run target(tid) on n_threads workers; return any exceptions raised."""
    errors: list[tuple[int, BaseException]] = []
    lock = threading.Lock()

    def runner(tid: int) -> None:
        try:
            target(tid)
        except BaseException as e:
            with lock:
                errors.append((tid, e))

    threads = [threading.Thread(target=runner, args=(i,)) for i in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return errors


class TestConcurrentOps:
    @pytest.mark.parametrize('num_threads,rows_per_thread', [(4, 100)])
    def test_concurrent_insert_and_select(self, uses_db: None, num_threads: int, rows_per_thread: int) -> None:
        """Verify that concurrent inserts and selects from multiple threads work correctly."""
        t = pxt.create_table(
            'test_concurrent',
            {'thread_id': pxt.Required[pxt.Int], 'row_idx': pxt.Required[pxt.Int], 'value': pxt.Required[pxt.Int]},
        )
        t.add_computed_column(doubled=t.value * 2)
        t.add_computed_column(offset=t.value + 100)

        errors: list[Exception] = []
        select_counts: list[int] = []
        lock = threading.Lock()

        def worker(tid: int) -> None:
            try:
                # each thread needs its own Table instance
                t = pxt.get_table('test_concurrent')
                for i in range(rows_per_thread):
                    status = t.insert([{'thread_id': tid, 'row_idx': i, 'value': tid * 1000 + i}])
                    validate_update_status(status, expected_rows=1)
                    if (i + 1) % 5 == 0:
                        rows = t.select().collect()
                        with lock:
                            select_counts.append(len(rows))
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == [], f'Worker threads raised exceptions: {errors}'

        result = t.select(t.thread_id, t.row_idx, t.value, t.doubled, t.offset).collect()
        assert len(result) == num_threads * rows_per_thread

        for row in result:
            assert row['doubled'] == row['value'] * 2
            assert row['offset'] == row['value'] + 100

        assert all(c >= 0 for c in select_counts)


class TestCrossThreadCatalog:
    """Catalog-bound objects (Query, Table, ColumnRef) constructed on the main thread
    must work safely when used from worker threads. Each thread has its own Catalog
    instance with its own TableVersion / sa_tbl per logical table; the per-instance
    _tbl_version cache on TVHs/TVPs and the cached `col` field on ColumnRefs were
    leaking the originating thread's sa_tbl into the worker's query plan and
    producing FROM-clause sa_tbl mismatches (DuplicateAlias) on master.
    """

    NUM_THREADS = 4
    ITERATIONS = 20

    def test_query_collected_from_workers(self, uses_db: None) -> None:
        """T1: Query built on main thread, collect() from worker threads."""
        t = pxt.create_table('t1', {'a': pxt.Required[pxt.Int], 'b': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': i, 'b': i * 10} for i in range(100)]), expected_rows=100)

        q = t.where(t.a >= 50).select(t.a, t.b)

        def worker(_tid: int) -> None:
            for _ in range(self.ITERATIONS):
                rows = q.collect()
                assert len(rows) == 50
                assert all(row['a'] >= 50 for row in rows)
                assert all(row['b'] == row['a'] * 10 for row in rows)

        errors = _run_workers(worker, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

    def test_dataframe_chained_on_main_executed_on_workers(self, uses_db: None) -> None:
        """T2: richer Expr tree (arithmetic, ordering) built on main, executed by workers."""
        t = pxt.create_table('t2', {'a': pxt.Required[pxt.Int], 'b': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': i, 'b': i * 10} for i in range(100)]), expected_rows=100)

        q = t.where(t.a > 50).select(sum_ab=t.a + t.b, double_a=t.a * 2).order_by(t.a)

        def worker(_tid: int) -> None:
            for _ in range(self.ITERATIONS):
                rows = q.collect()
                assert len(rows) == 49
                # rows are ordered by a; first row has a=51, b=510
                assert rows[0]['double_a'] == 102
                assert rows[0]['sum_ab'] == 51 + 510

        errors = _run_workers(worker, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

    def test_table_held_across_threads(self, uses_db: None) -> None:
        """T3: Table held on main, queries built and executed on workers."""
        t = pxt.create_table('t3', {'a': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': i} for i in range(100)]), expected_rows=100)

        def worker(_tid: int) -> None:
            for _ in range(self.ITERATIONS):
                rows = t.select(t.a).collect()
                assert len(rows) == 100

        errors = _run_workers(worker, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

    def test_column_ref_held_across_threads(self, uses_db: None) -> None:
        """T4: ColumnRefs captured on main are mixed into queries built on workers
        from a freshly fetched Table. This is the cleanest demonstration: FROM
        comes from worker's Table; SELECT references main's col_handle."""
        t = pxt.create_table('t4', {'a': pxt.Required[pxt.Int], 'b': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': i, 'b': i * 10} for i in range(100)]), expected_rows=100)
        a_ref = t.a
        b_ref = t.b

        def worker(_tid: int) -> None:
            t_worker = pxt.get_table('t4')
            for _ in range(self.ITERATIONS):
                rows = t_worker.select(a_ref, b_ref).collect()
                assert len(rows) == 100

        errors = _run_workers(worker, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

    def test_view_query_cross_thread(self, uses_db: None) -> None:
        """T5: query over a view, exercising the view+base TVP chain across threads."""
        t = pxt.create_table('t5_base', {'a': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': i} for i in range(100)]), expected_rows=100)
        v = pxt.create_view('t5_view', t.where(t.a >= 50))

        q = v.select(v.a)

        def worker(_tid: int) -> None:
            for _ in range(self.ITERATIONS):
                rows = q.collect()
                assert len(rows) == 50

        errors = _run_workers(worker, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

    def test_iterator_view_query_cross_thread(self, uses_db: None) -> None:
        """T6: query over a component (iterator) view across threads. Iterator views
        carry both base TVH and view TVH, plus the iterator function's own catalog
        references — the most TVH instances in a single plan."""
        t = pxt.create_table('t6_base', {'n': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'n': 3}, {'n': 5}]), expected_rows=2)
        iv = pxt.create_view('t6_iter_view', t, iterator=DummyIterator(t.n))

        q = iv.select(iv.out1, iv.out2)

        def worker(_tid: int) -> None:
            for _ in range(self.ITERATIONS):
                rows = q.collect()
                assert len(rows) == 3 + 5

        errors = _run_workers(worker, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'


class TestCrossXactStaleness:
    """ColumnRef.col was a cached field on master, so reads through a captured
    ColumnRef returned the pre-change Column instance after a schema change in
    a separate transaction. ColumnRef.col is now a property over col_handle and
    re-resolves through the catalog on every access.
    """

    def test_query_after_column_drop(self, uses_db: None) -> None:
        """T7: Query captured pre-drop, drop_column, re-execute. Should fail cleanly
        rather than crashing on a stale sa_col or silently returning bad rows."""
        t = pxt.create_table('t7', {'a': pxt.Required[pxt.Int], 'b': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': i, 'b': i * 10} for i in range(10)]), expected_rows=10)
        q = t.select(t.a, t.b)
        assert len(q.collect()) == 10  # pre-drop sanity check

        t.drop_column('b')

        # On loadtest, col_handle.get() raises NotFoundError because the dropped
        # col_id is no longer in cols_by_id. On master, ColumnRef.col is the
        # cached pre-drop Column and the plan references its stale sa_col.
        with pytest.raises(excs.Error):
            q.collect()

    def test_column_ref_attr_after_drop_and_readd(self, uses_db: None) -> None:
        """T8: ColumnRef captured pre-change, drop+re-add the column. Accessing
        the captured ref should re-resolve through the catalog (and raise, since
        the original col_id is gone) rather than returning stale Column metadata."""
        # `keep` exists so dropping 'a' is allowed (tables need at least one column).
        # The new 'a' is nullable since add_column on a non-empty table can't add a
        # required column.
        t = pxt.create_table('t8', {'a': pxt.Required[pxt.Int], 'keep': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': 1, 'keep': 0}]), expected_rows=1)

        a_ref = t.a
        # The Expr's col_type is set at __init__ from col.col_type; that snapshot
        # is fine. The bug is downstream — accessing ColumnRef.col on master
        # returned the cached Column instance forever.
        assert a_ref.col_type.is_int_type()

        t.drop_column('a')
        t.add_column(a=pxt.String)

        # On loadtest, .col is a property → col_handle.get() → raises NotFoundError
        # because the captured col_id was dropped. On master, .col is a cached
        # field and silently returns the stale Column.
        with pytest.raises(excs.Error):
            _ = a_ref.col
