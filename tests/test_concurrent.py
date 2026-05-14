import threading
import time
from typing import Callable

import pytest

import pixeltable as pxt
from tests.utils import DummyIterator, pxt_raises, validate_update_status


@pxt.udf
def _make_label(s: str) -> str:
    return f'lbl_{s}'


@pxt.udf
def _combine(n: int, double: int, s: str) -> str:
    return f'{s}:{n}:{double}'


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

    NUM_THREADS = 4
    ITERATIONS = 20

    def test_shared_query(self, uses_db: None) -> None:
        t = pxt.create_table('t1', {'a': pxt.Required[pxt.Int], 'b': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': i, 'b': i * 10} for i in range(100)]), expected_rows=100)

        q = t.where(t.a >= 50).select(t.a, t.b)

        def worker1(_tid: int) -> None:
            for _ in range(self.ITERATIONS):
                rows = q.collect()
                assert len(rows) == 50
                assert all(row['a'] >= 50 for row in rows)
                assert all(row['b'] == row['a'] * 10 for row in rows)

        errors = _run_workers(worker1, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

        q = t.where(t.a > 50).select(sum_ab=t.a + t.b, double_a=t.a * 2).order_by(t.a)

        def worker2(_tid: int) -> None:
            for _ in range(self.ITERATIONS):
                rows = q.collect()
                assert len(rows) == 49
                # rows are ordered by a; first row has a=51, b=510
                assert rows[0]['double_a'] == 102
                assert rows[0]['sum_ab'] == 51 + 510

        errors = _run_workers(worker2, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

    def test_shared_query_extended(self, uses_db: None) -> None:
        """A Query built on the main thread can be extended on a worker thread via builder methods."""
        t = pxt.create_table('t_ext', {'a': pxt.Required[pxt.Int], 'b': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': i, 'b': i * 10} for i in range(100)]), expected_rows=100)
        base = t.where(t.a >= 50).select(t.a, t.b)
        a_ref = t.a

        def worker(_tid: int) -> None:
            for _ in range(self.ITERATIONS):
                rows = base.order_by(a_ref).limit(10).collect()
                assert len(rows) == 10
                assert rows[0]['a'] == 50

        errors = _run_workers(worker, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

    def test_get_table(self, uses_db: None) -> None:
        t = pxt.create_table('t3', {'a': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': i} for i in range(100)]), expected_rows=100)

        def worker(_tid: int) -> None:
            t_worker = pxt.get_table('t3')
            for _ in range(self.ITERATIONS):
                rows = t_worker.select(t_worker.a).collect()
                assert len(rows) == 100

        errors = _run_workers(worker, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

    def test_shared_colrefs(self, uses_db: None) -> None:
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

    def test_shared_view_query1(self, uses_db: None) -> None:
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

    def test_shared_view_query2(self, uses_db: None) -> None:
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

    def test_non_collect_queries(self, uses_db: None) -> None:
        t = pxt.create_table('t10', {'a': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': i} for i in range(20)]), expected_rows=20)
        q = t.where(t.a >= 5).select(t.a)

        def worker(_tid: int) -> None:
            for _ in range(self.ITERATIONS):
                assert q.count() == 15
                assert len(q.head(5)) == 5
                assert len(q.show(5)) == 5
                assert len(q.tail(5)) == 5

        errors = _run_workers(worker, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

    def test_shared_query_udf(self, uses_db: None) -> None:
        t = pxt.create_table('t15', {'a': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': i} for i in range(50)]), expected_rows=50)

        @pxt.query
        def find_range(lower: int, upper: int) -> pxt.Query:
            return t.where((t.a >= lower) & (t.a <= upper)).select(t.a)

        # driver with varying centers; each [center-5, center+5] stays inside [0, 49] and contains
        # exactly 11 values so every per-row inner invocation has a deterministic length
        driver = pxt.create_table('t15_driver', {'center': pxt.Required[pxt.Int]})
        n_rows = 10
        validate_update_status(
            driver.insert([{'center': i % 40 + 5} for i in range(n_rows)]), expected_rows=n_rows
        )

        def worker(_tid: int) -> None:
            # pre-resolve both tables on this thread so they're catalog-warm before find_range's
            # inner invocation tries to load t inside the outer xact
            pxt.get_table('t15')
            driver_w = pxt.get_table('t15_driver')
            for _ in range(self.ITERATIONS):
                result = driver_w.select(rows=find_range(driver_w.center - 5, driver_w.center + 5)).collect()
                assert len(result) == n_rows
                assert all(len(result[i, 'rows']) == 11 for i in range(n_rows))

        errors = _run_workers(worker, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

    def test_concurrent_select_insert(self, uses_db: None) -> None:
        """
        Concurrent threads doing select and insert operations on the same table.

        Also exercises limit/offset under plan caching: a literal-limit/offset SqlNode path,
        and a query-template-with-Variable-limit path that flows through an SQL bindparam in
        LIMIT. Each reader iteration runs both shapes so plan invalidation across writer
        commits is exercised on plans that include limit/offset.

        TODO: programmatic validation of plan reuse (cache-hit count)
        """
        n0 = 20
        t = pxt.create_table(
            't_reader_writer',
            {'id': pxt.Required[pxt.Int], 'val': pxt.Required[pxt.String], 'n': pxt.Required[pxt.Int]},
        )
        t.add_computed_column(s_double=t.n * 2)
        t.add_computed_column(s_label=_make_label(t.val))
        t.add_computed_column(u_sum=t.n + t.s_double, stored=False)
        t.add_computed_column(u_check=_combine(t.n, t.s_double, t.val), stored=False)

        validate_update_status(t.insert([{'id': i, 'val': f'v{i}', 'n': i} for i in range(n0)]), expected_rows=n0)

        @pxt.query
        def top_ids(k: int) -> pxt.Query:
            return t.order_by(t.id).select(t.id).limit(k)

        # The writer sleeps longer than the reader between operations so the reader hits stretches where
        # the table version is unchanged and the cached plan is reused.

        n_writers = 2
        writes_per_writer = self.ITERATIONS
        reader_iterations = self.ITERATIONS * 4
        limit_k = 5
        offset_j = 3

        def writer(tid: int) -> None:
            t_w = pxt.get_table('t_reader_writer')
            for i in range(writes_per_writer):
                row_id = n0 + tid * writes_per_writer + i
                status = t_w.insert([{'id': row_id, 'val': f'v{row_id}', 'n': row_id}])
                validate_update_status(status, expected_rows=1)
                time.sleep(0.05)

        def reader(_tid: int) -> None:
            t_r = pxt.get_table('t_reader_writer')
            prev_len = 0
            prev_ids: set[int] = set()
            for _ in range(reader_iterations):
                rows = t_r.select(t_r.id, t_r.val, t_r.n, t_r.s_double, t_r.s_label, t_r.u_sum, t_r.u_check).collect()
                cur_len = len(rows)
                cur_ids = {row['id'] for row in rows}
                assert cur_len >= prev_len
                assert prev_ids.issubset(cur_ids)
                assert all(row['s_double'] == row['n'] * 2 for row in rows)
                assert all(row['s_label'] == f'lbl_{row["val"]}' for row in rows)
                assert all(row['u_sum'] == row['n'] + row['s_double'] for row in rows)
                assert all(row['u_check'] == f'{row["val"]}:{row["n"]}:{row["s_double"]}' for row in rows)
                prev_len = cur_len
                prev_ids = cur_ids

                # Literal limit/offset
                limited_rows = (
                    t_r.order_by(t_r.id).select(t_r.id, t_r.s_double).limit(limit_k, offset=offset_j).collect()
                )
                assert len(limited_rows) <= limit_k
                expected_ids = sorted(cur_ids)[offset_j : offset_j + limit_k]
                assert [row['id'] for row in limited_rows] == expected_ids
                assert all(row['s_double'] == row['id'] * 2 for row in limited_rows)

                # Variable limit via query template
                template_rows = t_r.select(top_k=top_ids(limit_k)).limit(1).collect()
                assert len(template_rows) == 1
                top_k_result = template_rows[0]['top_k']
                assert len(top_k_result) == min(limit_k, cur_len)
                assert [row['id'] for row in top_k_result] == sorted(cur_ids)[:limit_k]

                time.sleep(0.005)

        errors: list[tuple[int, BaseException]] = []
        lock = threading.Lock()

        def runner(role: str, tid: int, fn: Callable[[int], None]) -> None:
            try:
                fn(tid)
            except BaseException as e:
                with lock:
                    errors.append((tid, e))

        threads = [threading.Thread(target=runner, args=('w', i, writer)) for i in range(n_writers)]
        threads += [threading.Thread(target=runner, args=('r', i, reader)) for i in range(self.NUM_THREADS)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == [], f'errors: {errors[:3]}'
        assert t.count() == n0 + n_writers * writes_per_writer

    def test_shared_join(self, uses_db: None) -> None:
        t1 = pxt.create_table('t17_a', {'id': pxt.Required[pxt.Int], 'i': pxt.Required[pxt.Int]})
        t2 = pxt.create_table('t17_b', {'id': pxt.Required[pxt.Int], 'f': pxt.Required[pxt.Float]})
        validate_update_status(t1.insert([{'id': i, 'i': i} for i in range(20)]), expected_rows=20)
        validate_update_status(t2.insert([{'id': i, 'f': i * 1.5} for i in range(20)]), expected_rows=20)

        q = t1.join(t2, on=t1.id == t2.id, how='inner').select(t1.i, t2.f, out=t1.i + t2.f).order_by(t1.i)

        def worker(_tid: int) -> None:
            for _ in range(self.ITERATIONS):
                rows = q.collect()
                assert len(rows) == 20
                assert all(row['out'] == row['i'] + row['f'] for row in rows)

        errors = _run_workers(worker, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

    def test_shared_join2(self, uses_db: None) -> None:
        t1 = pxt.create_table('j1', {'id': pxt.Required[pxt.Int]})
        t2 = pxt.create_table('j2', {'id': pxt.Required[pxt.Int]})
        validate_update_status(t1.insert([{'id': i} for i in range(5)]), expected_rows=5)
        validate_update_status(t2.insert([{'id': i} for i in range(5)]), expected_rows=5)

        def worker(_tid: int) -> None:
            t1_worker = pxt.get_table('j1')

            # re-using t2 from the main thread raises
            with pxt_raises(pxt.ErrorCode.INVALID_STATE, match='thread'):
                t1_worker.join(t2, on=t1_worker.id == t2.id)
            with pxt_raises(pxt.ErrorCode.INVALID_STATE, match='thread'):
                t1_worker.select().join(t2, on=t1_worker.id == t2.id)

        errors = _run_workers(worker, n_threads=1)
        assert errors == [], f'worker raised: {errors[0][1]!r}'

    def test_shared_snapshot_query(self, uses_db: None) -> None:
        t = pxt.create_table('t19_base', {'a': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': i} for i in range(20)]), expected_rows=20)
        s = pxt.create_snapshot('t19_snap', t)

        q = s.where(s.a >= 10).select(s.a)

        def worker(_tid: int) -> None:
            for _ in range(self.ITERATIONS):
                assert len(q.collect()) == 10

        errors = _run_workers(worker, n_threads=self.NUM_THREADS)
        assert errors == [], f'errors: {errors[:3]}'

    def test_table_public_methods(self, uses_db: None) -> None:
        """All public Table methods guard against cross-thread calls."""
        t = pxt.create_table('t_xthread', {'a': pxt.Required[pxt.Int], 'keep': pxt.Required[pxt.Int]})
        validate_update_status(t.insert([{'a': 1, 'keep': 1}]), expected_rows=1)
        a_plus_one = t.a + 1

        def expect_error(call: Callable[[], object]) -> None:
            def worker(_tid: int) -> None:
                with pxt_raises(pxt.ErrorCode.INVALID_STATE, match='thread'):
                    call()

            errors = _run_workers(worker, n_threads=1)
            assert errors == [], f'worker raised: {errors[0][1]!r}'

        # Read / introspection
        expect_error(lambda: t.get_metadata())
        expect_error(lambda: t.list_views())
        expect_error(lambda: t.columns())
        expect_error(lambda: t.describe())
        expect_error(lambda: t.get_versions())
        expect_error(lambda: t.history())
        expect_error(lambda: t.external_stores())
        expect_error(lambda: t.get_base_table())

        # Query-builder entry points
        expect_error(lambda: t.select(t.a))
        expect_error(lambda: t.where(t.a > 0))
        expect_error(lambda: t.order_by(t.a))
        expect_error(lambda: t.limit(10))
        expect_error(lambda: t.distinct())
        expect_error(lambda: t.group_by(t.a))
        expect_error(lambda: t.sample(n=1))

        # Query terminals
        expect_error(lambda: t.collect())
        expect_error(lambda: t.count())
        expect_error(lambda: t.head(1))
        expect_error(lambda: t.tail(1))
        expect_error(lambda: t.show(1))
        expect_error(lambda: t.cursor())

        # Schema
        expect_error(lambda: t.add_column(b=pxt.String))
        expect_error(lambda: t.add_columns({'c': pxt.String}))
        expect_error(lambda: t.add_computed_column(double=a_plus_one))
        expect_error(lambda: t.drop_column('a'))
        expect_error(lambda: t.rename_column('a', 'aa'))

        # DML
        expect_error(lambda: t.insert([{'a': 2, 'keep': 1}]))
        expect_error(lambda: t.update({'a': a_plus_one}))
        expect_error(lambda: t.batch_update([{'a': 99, 'keep': 99}]))
        expect_error(lambda: t.delete())
        expect_error(lambda: t.recompute_columns('a'))

        # Versioning + attribute access
        expect_error(lambda: t.revert())
        expect_error(lambda: t.a)

        # Sanity: Table is still usable on the main thread.
        assert t.count() == 1
