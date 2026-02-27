import threading

import pytest

import pixeltable as pxt
from tests.utils import validate_update_status


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
