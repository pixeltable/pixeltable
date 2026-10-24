"""The storage engine (segments, history files, the primary-key index): every query returns the same results after a
reload and in another process, primary-key lookups agree with scans, and stores of another format are refused."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable._testing import reload_catalog


def raises(code: pxt.ErrorCode, match: str | None = None) -> pytest.RaisesExc:
    return pytest.raises(excs._error_class(code), match=match)


def run_python(code: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Runs `code` in a fresh interpreter (on the same store as this process unless `env` says otherwise)."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    return subprocess.run(
        [sys.executable, '-c', textwrap.dedent(code)], env=env or dict(os.environ), cwd=cwd, capture_output=True,
        text=True, timeout=300,
    )


def snapshot_results() -> dict[str, Any]:
    """The results of queries over every version of the tables `t` and `v` and the snapshot `s`."""
    out: dict[str, Any] = {}
    t = pxt.get_table('t')
    for version in range(t.get_metadata()['version'] + 1):
        tv = pxt.get_table(f't:{version}')
        out[f't:{version}'] = sorted((r['id'], r['val'], r['s']) for r in tv.select(tv.id, tv.val, tv.s).collect())
        out[f't:{version}:count'] = tv.count()
        out[f't:{version}:pk'] = [r['val'] for r in tv.where(tv.id.isin([7, 8])).select(tv.val).collect()]
    v = pxt.get_table('v')
    out['v'] = sorted((r['id'], r['d']) for r in v.select(v.id, v.d).collect())
    out['v:count'] = v.count()
    s = pxt.get_table('s')
    out['s'] = sorted(r['id'] for r in s.select(s.id).collect())
    out['versions'] = [(x['version'], x['inserts'], x['updates'], x['deletes']) for x in t.get_versions()]
    return out


def build_history() -> None:
    t = pxt.create_table('t', {'id': pxt.Int, 'val': pxt.Int, 's': pxt.String}, primary_key='id')
    pxt.create_view('v', t.where(t.val % 2 == 0), additional_columns={'d': t.val * 2})
    for b in range(30):
        t.insert({'id': i, 'val': i, 's': f'row {i}'} for i in range(b * 10, b * 10 + 10))
        if b % 3 == 0:
            t.update({'val': t.val + 1}, where=(t.id >= b * 5) & (t.id < b * 5 + 4))
        if b % 4 == 1:
            t.delete(where=t.id == b * 7)
        if b == 10:
            pxt.create_snapshot('s', t)
    t.batch_update([{'id': 9, 'val': -9}, {'id': 8, 'val': -8}])
    t.add_column(extra=pxt.Int | None)
    t.update({'extra': 1}, where=t.id < 50)
    t.revert()
    t.insert([{'id': 1000, 'val': 1000, 's': 'late'}])
    t.revert()
    t.delete(where=t.id > 250)


class TestReload:
    def test_queries_agree_after_reload(self) -> None:
        build_history()
        before = snapshot_results()
        reload_catalog()
        assert snapshot_results() == before
        # and after more commits on the reloaded catalog
        t = pxt.get_table('t')
        t.insert({'id': i, 'val': i, 's': 'x'} for i in range(2000, 2100))
        after = snapshot_results()
        reload_catalog()
        assert snapshot_results() == after

    def test_another_process_sees_the_same_results(self) -> None:
        build_history()
        expected = snapshot_results()
        r = run_python(
            """
            import json, sys
            sys.path.insert(0, '.')
            from test_storage import snapshot_results
            print(json.dumps(snapshot_results()))
            """
        )
        assert r.returncode == 0, r.stderr[-2000:]
        assert json.loads(r.stdout.strip().splitlines()[-1]) == json.loads(json.dumps(expected))

    def test_commits_of_another_process_are_picked_up(self) -> None:
        t = pxt.create_table('t', {'id': pxt.Int, 'val': pxt.Int}, primary_key='id')
        t.insert({'id': i, 'val': i} for i in range(100))
        assert t.count() == 100  # caches the rows
        r = run_python(
            """
            import pixeltable as pxt
            t = pxt.get_table('t')
            for b in range(10):
                t.insert({'id': i, 'val': i} for i in range(100 + b * 10, 110 + b * 10))
            t.delete(where=t.id < 10)
            t.update({'val': -1}, where=t.id == 50)
            """
        )
        assert r.returncode == 0, r.stderr[-2000:]
        assert t.count() == 190
        assert t.where(t.id == 50).select(t.val).collect()['val'] == [-1]
        assert t.where(t.id == 5).count() == 0
        assert pxt.get_table('t:1').count() == 100
        t.insert([{'id': 1000, 'val': 0}])
        r = run_python(
            """
            import pixeltable as pxt
            t = pxt.get_table('t')
            assert t.count() == 191, t.count()
            assert t.where(t.id == 1000).count() == 1
            """
        )
        assert r.returncode == 0, r.stderr[-2000:]

    def test_older_store_format_is_refused(self) -> None:
        home = tempfile.mkdtemp(prefix='pxtrs-old-')
        db_dir = os.path.join(home, 'pxtrs', 'pixeltable')
        os.makedirs(os.path.join(db_dir, 'tables'))
        with open(os.path.join(db_dir, 'catalog.json'), 'w') as f:
            json.dump({'format': 'pixeltable-rs', 'format_version': 1, 'catalog_id': '00000000-0000-0000-0000-000000000000',
                       'generation': 1, 'dirs': [], 'tables': []}, f)
        env = dict(os.environ, PIXELTABLE_HOME=home)
        env.pop('PIXELTABLE_DB', None)
        r = run_python(
            """
            import pixeltable as pxt
            try:
                pxt.list_tables()
            except pxt.Error as e:
                print(type(e).__name__, e)
            """,
            env=env,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        assert 'has format pixeltable-rs v1, which this version of pixeltable-rs (format v4) cannot read' in r.stdout


class TestPrimaryKeyIndex:
    def test_point_lookups_agree_with_scans(self) -> None:
        t = pxt.create_table('t', {'k': pxt.String, 'n': pxt.Int, 'v': pxt.Int}, primary_key=['k', 'n'])
        long_a = 'a' * 300 + 'x'
        long_b = 'a' * 300 + 'y'
        t.insert([{'k': 'p', 'n': 1, 'v': 1}, {'k': 'p', 'n': 2, 'v': 2}, {'k': 'q', 'n': 1, 'v': 3}])
        assert t.where((t.k == 'p') & (t.n == 2)).select(t.v).collect()['v'] == [2]
        assert t.where((t.n == 1) & (t.k == 'p') & (t.v > 0)).select(t.v).collect()['v'] == [1]
        assert t.where((t.k == 'p') & (t.n == 2) & (t.v > 5)).count() == 0
        assert sorted(t.where(t.k.isin(['p', 'q']) & (t.n == 1)).select(t.v).collect()['v']) == [1, 3]
        assert t.where((t.k == 'zz') & (t.n == 1)).count() == 0
        # strings that agree on the prefix the index of a data-versioned table keeps
        with raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key value'):
            t.insert([{'k': long_a, 'n': 1, 'v': 4}, {'k': long_b, 'n': 1, 'v': 5}])
        t.insert([{'k': long_a, 'n': 1, 'v': 4}])
        assert t.where((t.k == long_a) & (t.n == 1)).select(t.v).collect()['v'] == [4]
        assert t.where((t.k == long_b) & (t.n == 1)).count() == 0
        # a deleted key can be inserted again; old versions still find the old row
        v_before = t.get_metadata()['version']
        t.delete(where=(t.k == 'p') & (t.n == 1))
        assert t.where((t.k == 'p') & (t.n == 1)).count() == 0
        t.insert([{'k': 'p', 'n': 1, 'v': 10}])
        with raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key value'):
            t.insert([{'k': 'p', 'n': 1, 'v': 11}])
        old = pxt.get_table(f't:{v_before}')
        assert old.where((old.k == 'p') & (old.n == 1)).select(old.v).collect()['v'] == [1]
        assert t.where((t.k == 'p') & (t.n == 1)).select(t.v).collect()['v'] == [10]
        t.update({'v': 20}, where=(t.k == 'p') & (t.n == 1))
        assert t.where((t.k == 'p') & (t.n == 1)).select(t.v).collect()['v'] == [20]
        status = t.batch_update([{'k': 'q', 'n': 1, 'v': 30}, {'k': 'p', 'n': 2, 'v': 31}])
        assert status.num_rows == 2
        with raises(pxt.ErrorCode.ROW_NOT_FOUND):
            t.batch_update([{'k': 'nope', 'n': 1, 'v': 0}])
        t.revert()
        assert t.where((t.k == 'q') & (t.n == 1)).select(t.v).collect()['v'] == [3]
        reload_catalog()
        t = pxt.get_table('t')
        assert sorted(t.select(t.v).collect()['v']) == [2, 3, 4, 20]
        assert t.where((t.k == 'p') & (t.n == 1)).select(t.v).collect()['v'] == [20]

    def test_indexed_column_lookups_agree_with_scans(self) -> None:
        t = pxt.create_table('t', {'id': pxt.Int, 'k': pxt.String, 'n': pxt.Int | None})
        t.add_btree_index('k')
        t.add_btree_index('n')
        t.insert({'id': i, 'k': f'key{i % 300}', 'n': i % 7 if i % 11 else None} for i in range(3000))
        v = pxt.create_view('v', t.where(t.id % 2 == 0), additional_columns={'m': t.n + 1})
        v.add_btree_index('m')
        v1 = t.get_metadata()['version']
        t.update({'k': 'changed'}, where=t.id < 20)
        t.delete(where=t.id >= 2990)

        def check(tbl: Any, col: str, value: Any) -> None:
            expected = sorted(r['id'] for r in tbl.select(tbl.id, x=tbl[col]).collect() if r['x'] == value)
            got = sorted(tbl.where(tbl[col] == value).select(tbl.id).collect()['id'])
            assert got == expected, (col, value)

        for value in ['key5', 'key299', 'changed', 'nope']:
            check(t, 'k', value)
            check(v, 'k', value)
        for value in [0, 3, 6]:
            check(t, 'n', value)
        for value in [1, 4]:
            check(v, 'm', value)
        old = pxt.get_table(f't:{v1}')
        check(old, 'k', 'key5')
        assert old.where(old.k == 'changed').count() == 0
        assert sorted(t.where(t.k.isin(['key1', 'key2'])).select(t.id).collect()['id']) == sorted(
            i for i in range(20, 2990) if i % 300 in (1, 2)
        )
        reload_catalog()
        t = pxt.get_table('t')
        check(t, 'k', 'key7')

    def test_operational_table(self) -> None:
        t = pxt.create_table('o', {'id': pxt.Int, 'v': pxt.String}, primary_key='id', _is_data_versioned=False)
        t.insert({'id': i, 'v': str(i)} for i in range(50))
        t.delete(where=t.id < 10)
        t.insert({'id': i, 'v': 'again'} for i in range(5))
        t.update({'v': 'u'}, where=t.id == 20)
        with raises(pxt.ErrorCode.CONSTRAINT_VIOLATION):
            t.insert([{'id': 20, 'v': 'dup'}])
        expected = sorted((r['id'], r['v']) for r in t.collect())
        assert t.where(t.id == 20).select(t.v).collect()['v'] == ['u']
        reload_catalog()
        t = pxt.get_table('o')
        assert sorted((r['id'], r['v']) for r in t.collect()) == expected
        assert t.count() == 45


class TestThreads:
    def test_readers_see_consistent_snapshots_while_a_writer_commits(self) -> None:
        import threading

        t = pxt.create_table('t', {'id': pxt.Int, 'b': pxt.Int}, primary_key='id')
        errors: list[BaseException] = []
        done = threading.Event()

        def writer() -> None:
            try:
                for b in range(40):
                    t.insert({'id': b * 25 + i, 'b': b} for i in range(25))
                    if b % 5 == 4:
                        t.update({'b': t.b + 100}, where=t.b == b)
            except BaseException as e:
                errors.append(e)
            finally:
                done.set()

        def reader() -> None:
            try:
                while not done.is_set():
                    n = t.count()
                    assert n % 25 == 0, n
                    rows = t.select(t.id, t.b).collect()
                    assert len(rows) % 25 == 0
                    k = len(rows) // 2
                    if rows:
                        assert t.where(t.id == k).count() == 1
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=300)
        assert not errors, errors
        assert t.count() == 1000
        assert t.where(t.b >= 100).count() == 8 * 25

    def test_threads_read_a_freshly_loaded_table(self) -> None:
        import threading

        t = pxt.create_table('t', {'id': pxt.Int, 's': pxt.String}, primary_key='id')
        t.insert({'id': i, 's': f'v{i}'} for i in range(70_000))
        reload_catalog()
        t = pxt.get_table('t')
        errors: list[BaseException] = []
        results: list[int] = []

        def reader(k: int) -> None:
            try:
                if k % 2:
                    results.append(len(t.where(t.id == 12_345 + k).collect()))
                else:
                    results.append(t.where(t.s == f'v{k}').count())
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(k,)) for k in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=300)
        assert not errors, errors
        assert results == [1] * 8
