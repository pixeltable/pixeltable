"""
Probe for nested and chained JsonMapper behavior (not part of the tracked suite).

The mapper is only reachable via the function form pxtf.map(expr, fn); there is no .map() method here. Each test
asserts the INTENDED result; both currently fail. Run with a timeout, since a chain of two mappers can deadlock
rather than raise:

    timeout 40 pytest tests/test_mapper_probe.py
"""

import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf

pytestmark = pytest.mark.local('nested/chained mapper probe')


class TestMapperProbe:
    def test_nested_mapper(self, uses_db: None) -> None:
        # a mapper inside another mapper's fn: the inner mapper maps over each element of the outer element
        t = pxt.create_table('mapper_probe_nested', {'jj': pxt.Json})
        t.insert([{'jj': [[1, 2], [3]]}])
        res = t.select(o=pxtf.map(t.jj, lambda outer: pxtf.map(outer, lambda inner: inner + 1))).collect()
        assert res['o'][0] == [[2, 3], [4]]

    def test_chained_mapper(self, uses_db: None) -> None:
        # one mapper applied to the result of another
        t = pxt.create_table('mapper_probe_chained', {'j': pxt.Json})
        t.insert([{'j': [1, -2, 3]}])
        res = t.select(o=pxtf.map(pxtf.map(t.j, lambda x: x + 1), lambda x: x * 10)).collect()
        assert res['o'][0] == [20, -10, 40]
