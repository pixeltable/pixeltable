"""Regression tests for robustness defects: deep nesting, integer overflow, panics, several processes sharing a store,
and errors that escaped the pixeltable exception classes."""

import os
import subprocess
import sys
import textwrap
from typing import Any

import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable._testing import reload_catalog

MAX_JSON_DEPTH = 500  # value::MAX_JSON_DEPTH
MAX_EXPR_DEPTH = 500  # expr::MAX_EXPR_DEPTH


def raises(code: pxt.ErrorCode, match: str | None = None) -> pytest.RaisesExc:
    return pytest.raises(excs._error_class(code), match=match)


def nested_list(depth: int) -> list:
    """A list nested `depth` levels deep (`[]` has depth 1)."""
    v: list = []
    for _ in range(depth - 1):
        v = [v]
    return v


def depth_of(v: Any) -> int:
    d = 0
    while isinstance(v, (list, dict)):
        d += 1
        children = v if isinstance(v, list) else list(v.values())
        v = children[0] if children else None
    return d


def run_python(code: str, timeout: float = 120) -> subprocess.CompletedProcess:
    """Runs `code` in a fresh interpreter on the same store as this process."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    return subprocess.run(
        [sys.executable, '-c', textwrap.dedent(code)], env=dict(os.environ), cwd=cwd, capture_output=True, text=True,
        timeout=timeout,
    )


class TestJsonDepth:
    def test_deep_values_reload(self) -> None:
        # serde_json's default limit (128) once made a table holding a 127-deep value unreadable after reload
        t = pxt.create_table('d', {'j': pxt.Json | None})
        depths = [127, 128, 129, MAX_JSON_DEPTH]
        t.insert({'j': nested_list(d)} for d in depths)
        d = {}
        for _ in range(MAX_JSON_DEPTH - 1):
            d = {'k': d}
        t.insert([{'j': d}])
        reload_catalog()
        t = pxt.get_table('d')
        assert [depth_of(v) for v in t.collect()['j']] == [*depths, MAX_JSON_DEPTH]

    def test_too_deep_value_is_refused(self) -> None:
        t = pxt.create_table('d', {'j': pxt.Json | None})
        t.insert([{'j': [1]}])
        for depth in [MAX_JSON_DEPTH + 1, 20_000]:
            with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Error in column j: value is nested too deeply'):
                t.insert([{'j': nested_list(depth)}])
        d: dict = {}
        for _ in range(MAX_JSON_DEPTH):
            d = {'k': d}
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='nested too deeply'):
            t.insert([{'j': d}])
        reload_catalog()
        assert pxt.get_table('d').count() == 1

    def test_very_deep_value_does_not_crash(self) -> None:
        # used to overflow the Rust stack (segfault) while converting the Python value
        r = run_python(
            """
            import pixeltable as pxt
            t = pxt.create_table('deep', {'j': pxt.Json | None}, if_exists='replace')
            v = []
            for _ in range(20000):
                v = [v]
            try:
                t.insert([{'j': v}])
            except pxt.RequestError as e:
                print('refused:', str(e)[:60])
            """
        )
        assert r.returncode == 0, r.stderr[-2000:]
        assert 'refused: Error in column j: value is nested too deeply' in r.stdout

    def test_deep_literals(self) -> None:
        t = pxt.create_table('d', {'a': pxt.Int})
        t.insert([{'a': 1}])
        assert depth_of(t.select(x=nested_list(MAX_JSON_DEPTH)).collect()['x'][0]) == MAX_JSON_DEPTH
        with raises(pxt.ErrorCode.INVALID_ARGUMENT, match='nested too deeply'):
            t.select(x=nested_list(MAX_JSON_DEPTH + 1))
        with raises(pxt.ErrorCode.INVALID_ARGUMENT, match='nested too deeply'):
            t.select(x=nested_list(20_000))
        with raises(pxt.ErrorCode.INVALID_ARGUMENT, match='maximum supported dimension for an ndarray is currently 64'):
            pxt.array(nested_list(20_000))


class TestExprDepth:
    def test_deep_computed_column_reloads(self) -> None:
        # a computed column 70 levels deep once made catalog.json (and so the whole database) unreadable
        t = pxt.create_table('t', {'a': pxt.Int})
        t.insert([{'a': 1}])
        e = t.a
        for _ in range(MAX_EXPR_DEPTH - 1):
            e = e + 1
        assert t.select(x=e).collect()['x'] == [MAX_EXPR_DEPTH]
        t.add_computed_column(c=e)
        reload_catalog()
        t = pxt.get_table('t')
        assert t.select(t.c).collect()['c'] == [MAX_EXPR_DEPTH]
        t.insert([{'a': 2}])
        assert t.order_by(t.a).select(t.c).collect()['c'] == [MAX_EXPR_DEPTH, MAX_EXPR_DEPTH + 1]

    def test_too_deep_expression_is_refused(self) -> None:
        t = pxt.create_table('t', {'a': pxt.Int})
        e = t.a
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Expression is nested too deeply'):
            for _ in range(100_000):
                e = e + 1
        assert t.select(x=e).collect()['x'] == []


class TestIntegerOverflow:
    @pytest.fixture
    def tbl(self) -> pxt.Table:
        t = pxt.create_table('adv', {'a': pxt.Int | None, 's': pxt.String | None})
        t.insert([{'a': -(2**63), 's': 'x'}, {'a': 2**63 - 1, 's': 'y'}])
        return t

    def test_overflow_raises(self, tbl: pxt.Table) -> None:
        t = tbl
        for expr in [t.a // -1, t.a + 1, t.a - 1, t.a * 2, -t.a, t.a**100]:
            with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='^bigint out of range$'):
                t.select(x=expr).collect()
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='^bigint out of range$'):
            t.where(t.a // -1 > 0).count()
        assert t.select(x=t.a % -1).collect()['x'] == [0, 0]
        assert t.count() == 2  # still usable

    def test_floor_div_and_pow(self) -> None:
        t = pxt.create_table('n', {'a': pxt.Int})
        t.insert([{'a': -7}, {'a': 7}, {'a': 1}, {'a': -1}])
        assert t.select(x=t.a // 2).collect()['x'] == [-4, 3, 0, -1]
        assert t.select(x=t.a // -2).collect()['x'] == [3, -4, -1, 0]
        assert t.where(t.a.isin([1, -1])).select(x=t.a ** (2**40 + 1)).collect()['x'] == [1, -1]

    def test_repeat_too_large(self, tbl: pxt.Table) -> None:
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='requested length too large'):
            tbl.select(x=tbl.s * (2**62)).collect()


class TestMultiProcess:
    def test_no_lost_updates(self) -> None:
        t = pxt.create_table('shared', {'a': pxt.Int})
        t.insert([{'a': 1}])
        assert t.count() == 1
        r = run_python(
            """
            import pixeltable as pxt
            t = pxt.get_table('shared')
            t.insert([{'a': 2}, {'a': 3}])
            pxt.create_table('only_b', {'x': pxt.Int})
            """
        )
        assert r.returncode == 0, r.stderr[-2000:]
        # this process's catalog is stale: the next operation must pick up the other process's commit
        assert t.count() == 3
        t.insert([{'a': 4}])
        assert sorted(t.collect()['a']) == [1, 2, 3, 4]
        assert sorted(pxt.list_tables()) == ['only_b', 'shared']
        reload_catalog()
        assert sorted(pxt.get_table('shared').collect()['a']) == [1, 2, 3, 4]
        assert sorted(pxt.list_tables()) == ['only_b', 'shared']

    def test_concurrent_inserts(self) -> None:
        pxt.create_table('c', {'p': pxt.Int, 'i': pxt.Int})
        code = """
            import sys
            import pixeltable as pxt
            t = pxt.get_table('c')
            for i in range(15):
                t.insert([{'p': int(sys.argv[1]), 'i': i}])
            """
        cwd = os.path.dirname(os.path.abspath(__file__))
        procs = [
            subprocess.Popen([sys.executable, '-c', textwrap.dedent(code), str(p)], env=dict(os.environ), cwd=cwd,
                             stderr=subprocess.PIPE, text=True)
            for p in range(4)
        ]
        for p in procs:
            _, err = p.communicate(timeout=300)
            assert p.returncode == 0, err[-2000:]
        t = pxt.get_table('c')
        assert t.count() == 60
        assert sorted((r['p'], r['i']) for r in t.collect()) == [(p, i) for p in range(4) for i in range(15)]


class TestErrorClasses:
    def test_limit_out_of_range(self) -> None:
        t = pxt.create_table('t', {'a': pxt.Int})
        t.insert([{'a': 1}])
        for n in [2**63, 2**64]:
            with raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r"'limit\(\)' parameter must be <= 9223372036854775807"):
                t.limit(n)
            with raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r"'limit\(\)' parameter must be <= 9223372036854775807"):
                t.select().head(n)
        with raises(pxt.ErrorCode.INVALID_ARGUMENT, match="'offset' parameter must be <= 9223372036854775807"):
            t.limit(1, offset=2**63)
        with raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r"'limit\(\)' parameter must be >= 0"):
            t.limit(-(2**64))
        assert len(t.limit(2**63 - 1).collect()) == 1

    def test_lone_surrogates(self) -> None:
        t = pxt.create_table('t', {'s': pxt.String | None, 'j': pxt.Json | None})
        msg = "'utf-8' codec can't encode character '\\\\ud800' in position 0: surrogates not allowed"
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Error in column s: ' + msg):
            t.insert([{'s': '\ud800'}])
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Error in column j: ' + msg):
            t.insert([{'j': {'k': '\ud800'}}])
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Error in column j: ' + msg):
            t.insert([{'j': {'\ud800': 1}}])
        with raises(pxt.ErrorCode.INVALID_ARGUMENT, match=msg):
            t.where(t.s == '\ud800').count()
        assert t.count() == 0
