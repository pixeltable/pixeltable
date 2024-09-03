import datetime

import numpy as np
import pandas as pd
import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs

from ..utils import skip_test_if_not_installed


class TestPandas:
    def test_pandas_types(self, reset_db) -> None:
        df = pd.DataFrame({
            'int_col': [1, 2],
            'float_col': [1.0, 2.0],
            'bool_col': [True, False],
            'str_col': ['a', 'b'],
            'dt_col': [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)],
            'aware_dt_col': [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)],
            'json_col_1': [[1, 2], [3, 4]],
            'json_col_2': [{'a': 1}, {'b': 2}],
            'array_col_1': [np.ndarray((1, 2), dtype=np.int64), np.ndarray((3, 2), dtype=np.int64)],
            'array_col_2': [np.ndarray((1, 2), dtype=np.int64), np.ndarray((3, 4), dtype=np.int64)],
            'array_col_3': [np.ndarray((1, 2), dtype=np.float32), np.ndarray((3, 4), dtype=np.float32)],
            'image_col': [PIL.Image.new('RGB', (100, 100)), PIL.Image.new('L', (100, 200))],
        })
        t = pxt.io.import_pandas('test_types', df)
        assert(t._schema == {
            'int_col': pxt.IntType(nullable=True),
            'float_col': pxt.FloatType(nullable=True),
            'bool_col': pxt.BoolType(nullable=True),
            'str_col': pxt.StringType(nullable=True),
            'dt_col': pxt.TimestampType(nullable=True),
            'aware_dt_col': pxt.TimestampType(nullable=True),
            'json_col_1': pxt.JsonType(nullable=True),
            'json_col_2': pxt.JsonType(nullable=True),
            'array_col_1': pxt.ArrayType(shape=(None, 2), dtype=pxt.IntType(), nullable=True),
            'array_col_2': pxt.ArrayType(shape=(None, None), dtype=pxt.IntType(), nullable=True),
            'array_col_3': pxt.ArrayType(shape=(None, None), dtype=pxt.FloatType(), nullable=True),
            'image_col': pxt.ImageType(width=100, nullable=True),
        })

    def test_pandas_csv(self, reset_db) -> None:
        from pixeltable.io import import_csv

        t1 = import_csv('online_foods', 'tests/data/datasets/onlinefoods.csv')
        assert t1.count() == 388
        assert t1._schema == {
            'Age': pxt.IntType(nullable=True),
            'Gender': pxt.StringType(nullable=True),
            'Marital_Status': pxt.StringType(nullable=True),
            'Occupation': pxt.StringType(nullable=True),
            'Monthly_Income': pxt.StringType(nullable=True),
            'Educational_Qualifications': pxt.StringType(nullable=True),
            'Family_size': pxt.IntType(nullable=True),
            'latitude': pxt.FloatType(nullable=True),
            'longitude': pxt.FloatType(nullable=True),
            'Pin_code': pxt.IntType(nullable=True),
            'Output': pxt.StringType(nullable=True),
            'Feedback': pxt.StringType(nullable=True),
            'Unnamed__12': pxt.StringType(nullable=True),
        }
        assert t1.select(t1.Age).limit(5).collect()['Age'][:5] == [20, 24, 22, 22, 22]

        t2 = import_csv('ibm', 'tests/data/datasets/classeurIBM.csv', primary_key='Date')
        assert t2.count() == 4263
        assert t2._schema == {
            'Date': pxt.StringType(nullable=False),  # Primary key is non-nullable
            'Open': pxt.FloatType(nullable=True),
            'High': pxt.FloatType(nullable=True),
            'Low': pxt.FloatType(nullable=True),
            'Close': pxt.FloatType(nullable=True),
            'Volume': pxt.IntType(nullable=True),
            'Adj_Close': pxt.FloatType(nullable=True),
        }

        t3 = import_csv('edge_cases', 'tests/data/datasets/edge-cases.csv', parse_dates=['ts', 'ts_n'])
        assert t3.count() == 4
        assert t3._schema == {
            'c__int': pxt.IntType(nullable=True),
            'float': pxt.FloatType(nullable=True),
            'float_n': pxt.FloatType(nullable=True),
            'bool': pxt.BoolType(nullable=True),
            'string': pxt.StringType(nullable=True),
            'string_n': pxt.StringType(nullable=True),
            'ts': pxt.TimestampType(nullable=True),
            'ts_n': pxt.TimestampType(nullable=True),
        }
        result_set = t3.collect()
        assert result_set['c__int'] == [2, 3, 5, 22]
        assert result_set['float'] == [1.7, 3.0, 6.2, -1.0]
        _assert_equals_with_nans(result_set['float_n'], [1.0, 5.0, float('nan'), 1.0])
        assert result_set['bool'] == [True, False, True, True]
        assert result_set['string'] == ['fish', 'cake', 'salad', 'egg']
        assert result_set['string_n'] == ['fish', 'cake', None, 'egg']
        assert result_set['ts'] == [datetime.datetime(2024, 5, n) for n in range(3, 7)]
        assert result_set['ts_n'] == [datetime.datetime(2024, 5, 3), None, None, datetime.datetime(2024, 5, 6)]

    def test_pandas_images(self, reset_db) -> None:
        skip_test_if_not_installed('boto3')  # This test relies on s3 URLs
        from pixeltable.io.pandas import import_csv

        # Test overriding string type to images
        t4 = import_csv(
            'images', 'tests/data/datasets/images.csv', schema_overrides={'image': pxt.ImageType(nullable=True)}
        )
        assert t4.count() == 4
        assert t4._schema == {'name': pxt.StringType(nullable=True), 'image': pxt.ImageType(nullable=True)}
        result_set = t4.select(t4.image.width).collect()
        assert result_set['width'] == [1024, 962, 1024, None]

    def test_pandas_excel(self, reset_db) -> None:
        skip_test_if_not_installed('openpyxl')
        from pixeltable.io.pandas import import_excel

        t4 = import_excel('fin_sample', 'tests/data/datasets/Financial Sample.xlsx')
        assert t4.count() == 700
        assert t4._schema['Date'] == pxt.TimestampType(nullable=True)
        entry = t4.limit(1).collect()[0]
        assert entry['Date'] == datetime.datetime(2014, 1, 1, 0, 0)

        t5 = import_excel('sale_data', 'tests/data/datasets/SaleData.xlsx')
        assert t5.count() == 45
        assert t5._schema['OrderDate'] == pxt.TimestampType(nullable=True)
        # Ensure valid mapping of 'NaT' -> None
        assert t5.collect()[43]['OrderDate'] is None

    def test_pandas_errors(self, reset_db) -> None:
        from pixeltable.io import import_csv

        with pytest.raises(excs.Error) as exc_info:
            _ = import_csv(
                'online_foods', 'tests/data/datasets/onlinefoods.csv', schema_overrides={'Non-Column': pxt.StringType(nullable=True)}
            )
        assert '`Non-Column` specified in `schema_overrides` does not exist' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = import_csv(
                'edge_cases', 'tests/data/datasets/edge-cases.csv', primary_key=['!!int', 'Non-Column']
            )
        assert 'Primary key column `Non-Column` does not exist' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = import_csv(
                # String with null values
                'edge_cases', 'tests/data/datasets/edge-cases.csv', primary_key='string#n'
            )
        assert 'Primary key column `string#n` cannot contain null values.' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = import_csv(
                # Timestamp with null values
                'edge_cases', 'tests/data/datasets/edge-cases.csv', primary_key='ts_n'
            )
        assert 'Primary key column `ts_n` cannot contain null values.' in str(exc_info.value)


def _assert_equals_with_nans(a: list[float], b: list[float]) -> bool:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        # Need special handling since x != y is True if x and y are both nan
        if np.isnan(x):
            assert np.isnan(y)
        else:
            assert x == y
    return True
