"""The test hook `pixeltable._testing.validate_store()` (`Catalog::validate_store()`): passes for a store that
operations left consistent, and reports files the manifests do not account for."""

import os
from pathlib import Path

import pytest

import pixeltable as pxt
import pixeltable._testing


def tables_dir() -> Path:
    return Path(os.environ['PIXELTABLE_HOME']) / 'pxtrs' / os.environ.get('PIXELTABLE_DB', 'pixeltable') / 'tables'


@pxt.udf
def plus1(x: int) -> int:
    return x + 1


def test_consistent_after_operations() -> None:
    t = pxt.create_table('t', {'id': pxt.Int, 's': pxt.String, 'v': pxt.Int | None}, primary_key='id')
    t.insert({'id': i, 's': 'x' * (i * 40), 'v': i} for i in range(30))
    t.add_computed_column(w=plus1(t.v))
    t.add_btree_index(t.v)
    v = pxt.create_view('v', t.where(t.id < 20), additional_columns={'z': pxt.Int | None})
    v.add_computed_column(u=v.w * 2)
    t.update({'v': t.v + 1}, where=t.id < 10)
    t.batch_update([{'id': 3, 'v': 100}])
    t.delete(where=t.id > 25)
    t.revert()
    v.update({'z': 7})
    ops = pxt.create_table('ops', {'a': pxt.Int}, _is_data_versioned=False)
    ops.insert({'a': i} for i in range(10))
    ops.update({'a': ops.a + 1})
    ops.delete(where=ops.a > 5)
    pixeltable._testing.validate_store()
    pixeltable._testing.reload_catalog()
    pixeltable._testing.validate_store()


def test_reports_unaccounted_files() -> None:
    t = pxt.create_table('t', {'x': pxt.Int})
    t.insert({'x': i} for i in range(10))
    pixeltable._testing.validate_store()
    stray = tables_dir() / f'{"0" * 32}.1.seg'
    stray.write_bytes(b'')
    with pytest.raises(AssertionError, match='belongs to no table'):
        pixeltable._testing.validate_store()
    stray.unlink()
    segments = sorted(tables_dir().glob('*.seg'))
    assert len(segments) > 0
    segments[0].unlink()
    with pytest.raises(AssertionError, match='is missing'):
        pixeltable._testing.validate_store()
