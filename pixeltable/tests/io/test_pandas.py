import datetime

import numpy as np

import pixeltable as pxt
from ..utils import skip_test_if_not_installed


class TestPandas:

    def test_pandas_csv(self, reset_db) -> None:
        from pixeltable.io import import_csv

        t1 = import_csv('online_foods', 'pixeltable/tests/data/datasets/onlinefoods.csv')
        assert t1.count() == 388
        assert t1.column_types() == {
            'Age': pxt.IntType(),
            'Gender': pxt.StringType(),
            'Marital_Status': pxt.StringType(),
            'Occupation': pxt.StringType(),
            'Monthly_Income': pxt.StringType(),
            'Educational_Qualifications': pxt.StringType(),
            'Family_size': pxt.IntType(),
            'latitude': pxt.FloatType(),
            'longitude': pxt.FloatType(),
            'Pin_code': pxt.IntType(),
            'Output': pxt.StringType(),
            'Feedback': pxt.StringType(),
            'Unnamed__12': pxt.StringType()
        }
        assert t1.select(t1.Age).limit(5).collect()['Age'][:5] == [20, 24, 22, 22, 22]

        t2 = import_csv('ibm', 'pixeltable/tests/data/datasets/classeurIBM.csv')
        assert t2.count() == 4263
        assert t2.column_types() == {
            'Date': pxt.StringType(),
            'Open': pxt.FloatType(),
            'High': pxt.FloatType(),
            'Low': pxt.FloatType(),
            'Close': pxt.FloatType(),
            'Volume': pxt.IntType(),
            'Adj_Close': pxt.FloatType()
        }

        t3 = import_csv(
            'edge_cases',
            'pixeltable/tests/data/datasets/edge-cases.csv',
            parse_dates=['ts', 'ts_n']
        )
        assert t3.count() == 4
        assert t3.column_types() == {
            'int': pxt.IntType(),
            'float': pxt.FloatType(),
            'float_n': pxt.FloatType(),
            'bool': pxt.BoolType(),
            'string': pxt.StringType(),
            'string_n': pxt.StringType(nullable=True),
            'ts': pxt.TimestampType(),
            'ts_n': pxt.TimestampType(nullable=True),
        }
        result_set = t3.collect()
        assert result_set['int'] == [2, 3, 5, 22]
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
            'images',
            'pixeltable/tests/data/datasets/images.csv',
            schema={'name': pxt.StringType(), 'image': pxt.ImageType(nullable=True)}
        )
        assert t4.count() == 4
        assert t4.column_types() == {
            'name': pxt.StringType(),
            'image': pxt.ImageType(nullable=True)
        }
        result_set = t4.select(t4.image.width).collect()
        assert result_set['width'] == [1024, 962, 1024, None]

    def test_pandas_excel(self, reset_db) -> None:
        skip_test_if_not_installed('openpyxl')
        from pixeltable.io.pandas import import_excel

        t4 = import_excel('fin_sample', 'pixeltable/tests/data/datasets/Financial Sample.xlsx')
        assert t4.count() == 700
        assert t4.column_types()['Date'] == pxt.TimestampType()
        entry = t4.df().limit(1).collect()[0]
        assert entry['Date'] == datetime.datetime(2014, 1, 1, 0, 0)

        t5 = import_excel('sale_data', 'pixeltable/tests/data/datasets/SaleData.xlsx')
        assert t5.count() == 45
        assert t5.column_types()['OrderDate'] == pxt.TimestampType(nullable=True)
        # Ensure valid mapping of 'NaT' -> None
        assert t5.df().collect()[43]['OrderDate'] is None


def _assert_equals_with_nans(a: list[float], b: list[float]) -> bool:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        # Need special handling since x != y is True if x and y are both nan
        if np.isnan(x):
            assert np.isnan(y)
        else:
            assert x == y
    return True
