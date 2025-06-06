import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.env import Env

from ..utils import skip_test_if_not_installed


class TestPandas:
    def make_src_data(self) -> dict[str, object]:
        src_data = {
            'int_col': [1, 2],
            'float_col': [1.0, 2.0],
            'bool_col': [True, False],
            'str_col': ['a', 'b'],
            'dt_col': [datetime.datetime(2024, 1, 1, 1, 1, 1), datetime.datetime(2024, 1, 2, 1, 1, 1)],
            'aware_dt_col': [
                datetime.datetime(2024, 1, 1, 1, 1, 1, tzinfo=(ZoneInfo('Europe/Moscow'))),
                datetime.datetime(2024, 1, 1, 1, 1, 1, tzinfo=datetime.timezone.utc),
            ],
            'date_col': [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
            'json_col_1': [[1, 2], [3, 4]],
            'json_col_2': [{'a': 1}, {'b': 2}],
            'array_col_1': [np.ndarray((1, 2), dtype=np.int64), np.ndarray((3, 2), dtype=np.int64)],
            'array_col_2': [np.ndarray((1, 2), dtype=np.int64), np.ndarray((3, 4), dtype=np.int64)],
            'array_col_3': [np.ndarray((1, 2), dtype=np.float32), np.ndarray((3, 4), dtype=np.float32)],
            'image_col': [PIL.Image.new('RGB', (100, 100)), PIL.Image.new('L', (100, 200))],
        }
        return src_data

    def test_import_pandas_types(self, reset_db: None) -> None:
        default_tz = Env.get().default_time_zone

        src_data = self.make_src_data()
        df = pd.DataFrame(src_data)

        t = pxt.io.import_pandas('test_types', df)
        assert t._schema() == {
            'int_col': ts.IntType(nullable=True),
            'float_col': ts.FloatType(nullable=True),
            'bool_col': ts.BoolType(nullable=True),
            'str_col': ts.StringType(nullable=True),
            'dt_col': ts.TimestampType(nullable=True),
            'aware_dt_col': ts.TimestampType(nullable=True),
            'date_col': ts.DateType(nullable=True),
            'json_col_1': ts.JsonType(nullable=True),
            'json_col_2': ts.JsonType(nullable=True),
            'array_col_1': ts.ArrayType(shape=(None, 2), dtype=ts.IntType(), nullable=True),
            'array_col_2': ts.ArrayType(shape=(None, None), dtype=ts.IntType(), nullable=True),
            'array_col_3': ts.ArrayType(shape=(None, None), dtype=ts.FloatType(), nullable=True),
            'image_col': ts.ImageType(width=100, nullable=True),
        }
        res = t.select().order_by(t.int_col).collect()
        assert res['int_col'] == src_data['int_col']
        assert res['float_col'] == src_data['float_col']
        assert res['bool_col'] == src_data['bool_col']
        assert res['str_col'] == src_data['str_col']
        assert res['dt_col'] == [
            datetime.datetime(2024, 1, 1, 1, 1, 1).astimezone(default_tz),
            datetime.datetime(2024, 1, 2, 1, 1, 1).astimezone(default_tz),
        ]
        assert res['aware_dt_col'] == [
            datetime.datetime(2024, 1, 1, 1, 1, 1, tzinfo=ZoneInfo('Europe/Moscow')).astimezone(default_tz),
            datetime.datetime(2024, 1, 1, 1, 1, 1, tzinfo=datetime.timezone.utc),
        ]
        assert res['date_col'] == src_data['date_col']
        assert res['json_col_1'] == src_data['json_col_1']
        assert res['json_col_2'] == src_data['json_col_2']
        assert t.count() == len(df)

    def test_insert_pandas_types(self, reset_db: None) -> None:
        src_data = self.make_src_data()
        df = pd.DataFrame(src_data)
        t = pxt.io.import_pandas('test_types', df)
        assert t._schema() == {
            'int_col': ts.IntType(nullable=True),
            'float_col': ts.FloatType(nullable=True),
            'bool_col': ts.BoolType(nullable=True),
            'str_col': ts.StringType(nullable=True),
            'dt_col': ts.TimestampType(nullable=True),
            'aware_dt_col': ts.TimestampType(nullable=True),
            'date_col': ts.DateType(nullable=True),
            'json_col_1': ts.JsonType(nullable=True),
            'json_col_2': ts.JsonType(nullable=True),
            'array_col_1': ts.ArrayType(shape=(None, 2), dtype=ts.IntType(), nullable=True),
            'array_col_2': ts.ArrayType(shape=(None, None), dtype=ts.IntType(), nullable=True),
            'array_col_3': ts.ArrayType(shape=(None, None), dtype=ts.FloatType(), nullable=True),
            'image_col': ts.ImageType(width=100, nullable=True),
        }
        assert t.count() == len(df)
        t.insert(df)
        assert t.count() == 2 * len(df)

    def test_import_pandas_csv(self, reset_db: None) -> None:
        from pixeltable.io import import_csv

        t1 = import_csv('online_foods', 'tests/data/datasets/onlinefoods.csv')
        assert t1.count() == 388
        assert t1._schema() == {
            'Age': ts.IntType(nullable=True),
            'Gender': ts.StringType(nullable=True),
            'Marital_Status': ts.StringType(nullable=True),
            'Occupation': ts.StringType(nullable=True),
            'Monthly_Income': ts.StringType(nullable=True),
            'Educational_Qualifications': ts.StringType(nullable=True),
            'Family_size': ts.IntType(nullable=True),
            'latitude': ts.FloatType(nullable=True),
            'longitude': ts.FloatType(nullable=True),
            'Pin_code': ts.IntType(nullable=True),
            'Output': ts.StringType(nullable=True),
            'Feedback': ts.StringType(nullable=True),
            'Unnamed__12': ts.StringType(nullable=True),
        }
        assert t1.select(t1.Age).limit(5).collect()['Age'][:5] == [20, 24, 22, 22, 22]

        t1a = pxt.create_table('online_foods_a', source='tests/data/datasets/onlinefoods.csv')
        assert t1a.count() == 388
        assert t1.show() == t1a.show()

        t1a.insert('tests/data/datasets/onlinefoods.csv')
        assert t1a.count() == 2 * 388

        t2 = import_csv('ibm', 'tests/data/datasets/classeurIBM.csv', primary_key='Date')
        assert t2.count() == 4263
        assert t2._schema() == {
            'Date': ts.StringType(nullable=False),  # Primary key is non-nullable
            'Open': ts.FloatType(nullable=True),
            'High': ts.FloatType(nullable=True),
            'Low': ts.FloatType(nullable=True),
            'Close': ts.FloatType(nullable=True),
            'Volume': ts.IntType(nullable=True),
            'Adj_Close': ts.FloatType(nullable=True),
        }

        t3 = import_csv('edge_cases', 'tests/data/datasets/edge-cases.csv', parse_dates=['ts', 'ts_n'])
        assert t3.count() == 4
        assert t3._schema() == {
            'c__int': ts.IntType(nullable=True),
            'float': ts.FloatType(nullable=True),
            'float_n': ts.FloatType(nullable=True),
            'bool': ts.BoolType(nullable=True),
            'string': ts.StringType(nullable=True),
            'string_n': ts.StringType(nullable=True),
            'ts': ts.TimestampType(nullable=True),
            'ts_n': ts.TimestampType(nullable=True),
        }
        result_set = t3.collect()
        assert result_set['c__int'] == [2, 3, 5, 22]
        assert result_set['float'] == [1.7, 3.0, 6.2, -1.0]
        _assert_equals_with_nans(result_set['float_n'], [1.0, 5.0, float('nan'), 1.0])
        assert result_set['bool'] == [True, False, True, True]
        assert result_set['string'] == ['fish', 'cake', 'salad', 'egg']
        assert result_set['string_n'] == ['fish', 'cake', None, 'egg']
        # Timestamps coming out of the DB will always be aware; we need to compare them to aware datetimes
        assert result_set['ts'] == [datetime.datetime(2024, 5, n).astimezone(None) for n in range(3, 7)]
        assert result_set['ts_n'] == [
            datetime.datetime(2024, 5, 3).astimezone(None),
            None,
            None,
            datetime.datetime(2024, 5, 6).astimezone(None),
        ]

    def test_insert_pandas_csv(self, reset_db: None) -> None:
        from pixeltable.io import import_csv

        t1 = import_csv('online_foods', 'tests/data/datasets/onlinefoods.csv')
        assert t1.count() == 388
        t1.insert('tests/data/datasets/onlinefoods.csv')
        assert t1.count() == 2 * 388

        t2 = import_csv('ibm', 'tests/data/datasets/classeurIBM.csv', primary_key='Date')
        assert t2.count() == 4263
        t2.insert('tests/data/datasets/classeurIBM.csv')
        assert t2.count() == 2 * 4263

        t3 = import_csv('edge_cases', 'tests/data/datasets/edge-cases.csv', parse_dates=['ts', 'ts_n'])
        assert t3.count() == 4
        t3.insert('tests/data/datasets/edge-cases.csv')
        assert t3.count() == 2 * 4

    def test_pandas_images(self, reset_db: None) -> None:
        skip_test_if_not_installed('boto3')  # This test relies on s3 URLs
        from pixeltable.io.pandas import import_csv

        # Test overriding string type to images
        t4 = import_csv(
            'images', 'tests/data/datasets/images.csv', schema_overrides={'image': ts.ImageType(nullable=True)}
        )
        assert t4.count() == 4
        assert t4._schema() == {'name': ts.StringType(nullable=True), 'image': ts.ImageType(nullable=True)}
        result_set = t4.order_by(t4.name).select(t4.image.width).collect()
        assert result_set['width'] == [1024, None, 1024, 962]

    def test_import_pandas_excel(self, reset_db: None) -> None:
        skip_test_if_not_installed('openpyxl')
        from pixeltable.io.pandas import import_excel

        t4 = import_excel('fin_sample', 'tests/data/datasets/Financial Sample.xlsx')
        assert t4.count() == 700
        assert t4._schema()['Date'] == ts.TimestampType(nullable=True)
        entry = t4.limit(1).collect()[0]
        assert entry['Date'] == datetime.datetime(2014, 1, 1, 0, 0).astimezone(None)

        t5 = import_excel('sale_data', 'tests/data/datasets/SaleData.xlsx')
        assert t5.count() == 45
        assert t5._schema()['OrderDate'] == ts.TimestampType(nullable=True)
        # Ensure valid mapping of 'NaT' -> None
        assert t5.collect()[43]['OrderDate'] is None

        t6 = import_excel('questions', 'docs/resources/rag-demo/Q-A-Rag.xlsx')
        assert t6.count() == 8
        # Ensure that StringType is used when the column contains mixed types
        assert t6._schema()['correct_answer'] == ts.StringType(nullable=True)

    def test_insert_pandas_excel(self, reset_db: None) -> None:
        skip_test_if_not_installed('openpyxl')
        from pixeltable.io.pandas import import_excel

        t4 = import_excel('fin_sample', 'tests/data/datasets/Financial Sample.xlsx')
        assert t4.count() == 700
        t4.insert('tests/data/datasets/Financial Sample.xlsx')
        assert t4.count() == 2 * 700

        t5 = import_excel('sale_data', 'tests/data/datasets/SaleData.xlsx')
        assert t5.count() == 45
        t5.insert('tests/data/datasets/SaleData.xlsx')
        assert t5.count() == 2 * 45

        t6 = import_excel('questions', 'docs/resources/rag-demo/Q-A-Rag.xlsx')
        assert t6.count() == 8
        t6.insert('docs/resources/rag-demo/Q-A-Rag.xlsx')
        assert t6.count() == 2 * 8

    def test_pandas_errors(self, reset_db: None) -> None:
        from pixeltable.io import import_csv

        with pytest.raises(excs.Error) as exc_info:
            _ = import_csv(
                'online_foods',
                'tests/data/datasets/onlinefoods.csv',
                schema_overrides={'Non-Column': ts.StringType(nullable=True)},
            )
        assert 'Some column(s) specified in `schema_overrides` are not present' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = import_csv('edge_cases', 'tests/data/datasets/edge-cases.csv', primary_key=['!!int', 'Non-Column'])
        assert 'Primary key column(s) are not found in the source:' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = import_csv(
                # String with null values
                'edge_cases',
                'tests/data/datasets/edge-cases.csv',
                primary_key='string#n',
            )
        assert 'Primary key column `string#n` cannot contain null values.' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = import_csv(
                # Timestamp with null values
                'edge_cases',
                'tests/data/datasets/edge-cases.csv',
                primary_key='ts_n',
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
