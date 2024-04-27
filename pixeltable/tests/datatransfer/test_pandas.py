import pixeltable as pxt


class TestPandas:

    def test_pandas(self, test_client: pxt.Client) -> None:
        cl = test_client
        t = cl.import_csv('online_foods', 'pixeltable/tests/data/datasets/onlinefoods.csv')
        assert t.count() == 388
        assert t.column_types() == {
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
        assert t.select(t.Age).limit(5).collect()['Age'][:5] == [20, 24, 22, 22, 22]
