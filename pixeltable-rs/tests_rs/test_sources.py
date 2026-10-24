"""Data sources of `create_table(source=...)` and `insert()`: pandas DataFrames and CSV/JSON files (upstream
`PandasTableDataConduit`, `CSVTableDataConduit`, `JsonTableDataConduit`)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs


def raises(code: pxt.ErrorCode, match: str | None = None) -> pytest.RaisesExc:
    return pytest.raises(excs._error_class(code), match=match)


class TestDataFrameSources:
    def test_create_and_insert(self) -> None:
        df = pd.DataFrame(
            {
                'Int Col': [1, 2, 3],
                'f': [1.5, np.nan, 3.0],
                'b': [True, False, True],
                'ts': pd.to_datetime(['2024-01-01', None, '2024-01-03']),
                'obj': [{'a': 1}, None, {'a': 2}],
                '1st': ['x', 'y', 'z'],
                'class': [1, 2, 3],
            }
        )
        t = pxt.create_table('df_src', source=df, primary_key='Int Col')
        types = {name: md['type_'] for name, md in t.get_metadata()['columns'].items()}
        # names are normalized to identifiers (keywords and reserved names get a suffix); non-key columns are nullable
        assert types == {
            'int_col': 'Int',
            'f': 'Float | None',
            'b': 'Bool | None',
            'ts': 'Timestamp | None',
            'obj': "Json[{'a': Int}] | None",
            'c_1st': 'String | None',
            'class_1': 'Int | None',
        }
        res = t.order_by(t.int_col).collect()
        assert res['obj'] == [{'a': 1}, None, {'a': 2}]
        assert res['ts'][1] is None

        status = t.insert(pd.DataFrame({'Int Col': [4], 'b': [False]}))
        assert status.num_rows == 1 and t.count() == 4
        with raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='Unknown column name zzz'):
            t.insert(pd.DataFrame({'zzz': [1]}))
        with raises(pxt.ErrorCode.MISSING_REQUIRED, match=r'Missing required column\(s\) \(int_col\)'):
            t.insert(pd.DataFrame({'f': [1.0]}))

    def test_schema_errors(self) -> None:
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Primary key column `a` cannot contain null values'):
            pxt.create_table('pk_nulls', source=pd.DataFrame({'a': [1, None]}), primary_key='a')
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='not present in the source: b'):
            pxt.create_table('extra_override', source=pd.DataFrame({'a': [1]}), schema_overrides={'b': pxt.Int})
        with raises(pxt.ErrorCode.INVALID_TYPE, match=r'Could not infer Pixeltable type of column: a \(dtype: Int64\)'):
            pxt.create_table('ext_dtype', source=pd.DataFrame({'a': pd.array([1, None], dtype='Int64')}))
        t = pxt.create_table('override', source=pd.DataFrame({'a': [1, 2]}), schema_overrides={'a': pxt.Float})
        assert t.get_metadata()['columns']['a']['type_'] == 'Float'


class TestFileSources:
    def test_csv_and_json(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'rows.csv'
        pd.DataFrame({'x': [1, 2], 'y': ['p', 'q']}).to_csv(csv_path, index=False)
        t = pxt.create_table('csv_src', source=str(csv_path))
        assert t.order_by(t.x).collect()['y'] == ['p', 'q']
        # extra_args go to pandas.read_csv()
        t2 = pxt.create_table('csv_args', source=str(csv_path), extra_args={'usecols': ['x']})
        assert list(t2.get_metadata()['columns']) == ['x']

        jsonl_path = tmp_path / 'rows.jsonl'
        jsonl_path.write_text(json.dumps({'x': 5, 'y': 'r'}) + '\n\n' + json.dumps({'x': 6, 'y': 's'}) + '\n')
        t.insert(str(jsonl_path))
        assert t.count() == 4
        bad_path = tmp_path / 'bad.jsonl'
        bad_path.write_text('{"x": 1}\nnot json\n')
        with raises(pxt.ErrorCode.INVALID_DATA_FORMAT, match='Invalid JSONL line 2'):
            t.insert(str(bad_path))

        json_path = tmp_path / 'rows.json'
        json_path.write_text(json.dumps([{'x': 7, 'y': 't'}]))
        t3 = pxt.create_table('json_src', source=str(json_path))
        assert t3.collect()['x'] == [7]

    def test_parquet(self, tmp_path: Path) -> None:
        import datetime

        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / 'rows.parquet'
        table = pa.table(
            {
                'i': pa.array([1, 2, None], type=pa.int32()),
                'f': pa.array([1.5, None, 2.5], type=pa.float64()),
                's': pa.array(['a', 'b', None]),
                'd': pa.array([datetime.date(2024, 1, 1), None, datetime.date(2024, 1, 3)]),
                'st': pa.array([{'x': 1}, None, {'x': 2}]),
            }
        )
        pq.write_table(table, path)
        t = pxt.create_table('pq_src', source=str(path))
        types = {name: md['type_'] for name, md in t.get_metadata()['columns'].items()}
        assert types == {'i': 'Int | None', 'f': 'Float | None', 's': 'String | None', 'd': 'Date | None', 'st': 'Json | None'}
        res = t.order_by(t.f).collect()
        assert res['st'] == [{'x': 1}, {'x': 2}, None]
        t.insert(str(path))
        assert t.count() == 6
        with raises(pxt.ErrorCode.INVALID_TYPE, match=r'Could not infer pixeltable type for column\(s\): n'):
            bad = tmp_path / 'bad.parquet'
            pq.write_table(pa.table({'n': pa.array([None, None], type=pa.null())}), bad)
            pxt.create_table('pq_bad', source=str(bad))
