import numpy as np

import pixeltable as pxt


class TestPandas:

    def test_pandas(self, test_client: pxt.Client) -> None:
        cl = test_client

        t1 = cl.import_csv('online_foods', 'pixeltable/tests/data/datasets/onlinefoods.csv')
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

        t2 = cl.import_csv('ibm', 'pixeltable/tests/data/datasets/classeurIBM.csv')
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

        t3 = cl.import_csv('edge_cases', 'pixeltable/tests/data/datasets/edge-cases.csv')
        assert t3.count() == 4
        assert t3.column_types() == {
            'col1': pxt.FloatType(),
            'col2': pxt.StringType(),
            'col3': pxt.FloatType(),
            'col4': pxt.FloatType()
        }
        result_set = t3.collect()
        _assert_equals_with_nans(result_set['col1'], [1.0, 5.0, float('nan'), 1.0])
        assert result_set['col2'] == ['fish', '7', 'nan', '3']
        _assert_equals_with_nans(result_set['col3'], [3.0, float('nan'), float('nan'), 5.0])
        _assert_equals_with_nans(result_set['col4'], [float('nan'), float('nan'), float('nan'), 7.0])

        t4 = cl.import_xlsx('')


def _assert_equals_with_nans(a: list[float], b: list[float]) -> bool:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        # Need special handling since x != y is True if x and y are both nan
        if np.isnan(x):
            assert np.isnan(y)
        else:
            assert x == y
    return True
