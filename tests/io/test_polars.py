import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.env import Env

from ..utils import skip_test_if_not_installed

# Import Polars conditionally for testing
pytest_plugins = []
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pytest.skip("Polars not available", allow_module_level=True)


class TestPolars:
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
            'array_col_1': [[1, 2], [3, 4]],  # Polars handles arrays as lists
            'array_col_2': [[1, 2, 3], [4, 5, 6]],  # Variable length arrays
        }
        return src_data

    def test_import_polars_types(self, reset_db: None) -> None:
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        default_tz = Env.get().default_time_zone

        src_data = self.make_src_data()
        df = pl.DataFrame(src_data)

        t = pxt.io.import_polars('test_types', df)
        
        # Verify the schema matches expected types
        expected_schema = {
            'int_col': ts.IntType(nullable=True),
            'float_col': ts.FloatType(nullable=True),
            'bool_col': ts.BoolType(nullable=True),
            'str_col': ts.StringType(nullable=True),
            'dt_col': ts.TimestampType(nullable=True),
            'aware_dt_col': ts.TimestampType(nullable=True),
            'date_col': ts.DateType(nullable=True),
            'json_col_1': ts.JsonType(nullable=True),  # Lists become JSON in Polars
            'json_col_2': ts.JsonType(nullable=True),
            'array_col_1': ts.ArrayType(shape=(None, 2), dtype=ts.IntType(), nullable=True),
            'array_col_2': ts.ArrayType(shape=(None, None), dtype=ts.IntType(), nullable=True),
        }
        
        actual_schema = t._get_schema()
        assert actual_schema == expected_schema
        
        # Verify data was imported correctly
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

    def test_insert_polars_types(self, reset_db: None) -> None:
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        src_data = self.make_src_data()
        df = pl.DataFrame(src_data)
        t = pxt.io.import_polars('test_types', df)
        
        initial_count = t.count()
        assert initial_count == len(df)
        
        # Insert the same DataFrame again
        t.insert(df)
        assert t.count() == 2 * initial_count

    def test_import_polars_csv(self, reset_db: None) -> None:
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        from pixeltable.io.polars import import_polars_csv

        t1 = import_polars_csv('online_foods', 'tests/data/datasets/onlinefoods.csv')
        assert t1.count() == 388
        
        # Verify basic schema structure (Polars may infer types differently than pandas)
        schema = t1._get_schema()
        assert 'Age' in schema
        assert 'Gender' in schema
        assert 'Output' in schema
        
        # Verify some data
        first_ages = t1.select(t1.Age).limit(5).collect()['Age'][:5]
        assert len(first_ages) == 5
        
        # Test insertion from the same CSV
        t1.insert('tests/data/datasets/onlinefoods.csv')
        assert t1.count() == 2 * 388

    def test_import_polars_parquet(self, reset_db: None) -> None:
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        # First create a test parquet file using Polars
        test_data = {
            'id': [1, 2, 3, 4],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'score': [95.5, 87.2, 92.1, 88.8],
            'active': [True, False, True, True]
        }
        
        df = pl.DataFrame(test_data)
        test_file = 'test_polars.parquet'
        df.write_parquet(test_file)
        
        try:
            from pixeltable.io.polars import import_polars_parquet
            
            t = import_polars_parquet('test_parquet', test_file)
            assert t.count() == 4
            
            # Verify schema
            schema = t._get_schema()
            assert schema['id'] == ts.IntType(nullable=True)
            assert schema['name'] == ts.StringType(nullable=True) 
            assert schema['score'] == ts.FloatType(nullable=True)
            assert schema['active'] == ts.BoolType(nullable=True)
            
            # Verify data
            res = t.select().order_by(t.id).collect()
            assert res['id'] == test_data['id']
            assert res['name'] == test_data['name']
            assert res['score'] == test_data['score']
            assert res['active'] == test_data['active']
            
        finally:
            # Clean up test file
            import os
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_polars_with_schema_overrides(self, reset_db: None) -> None:
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        # Create a DataFrame with a string column that we want to override as Image
        data = {
            'name': ['image1', 'image2'],
            'url': ['http://example.com/1.jpg', 'http://example.com/2.jpg'],
            'score': [0.95, 0.87]
        }
        df = pl.DataFrame(data)
        
        # Override the url column to be Image type
        t = pxt.io.import_polars(
            'test_overrides', 
            df, 
            schema_overrides={'url': pxt.Image}
        )
        
        schema = t._get_schema()
        assert schema['name'] == ts.StringType(nullable=True)
        assert schema['url'] == ts.ImageType(nullable=True)
        assert schema['score'] == ts.FloatType(nullable=True)

    def test_polars_with_primary_key(self, reset_db: None) -> None:
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        data = {
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com']
        }
        df = pl.DataFrame(data)
        
        t = pxt.io.import_polars('test_pk', df, primary_key='id')
        
        schema = t._get_schema()
        # Primary key columns should be non-nullable
        assert schema['id'] == ts.IntType(nullable=False)
        assert schema['name'] == ts.StringType(nullable=True)
        assert schema['email'] == ts.StringType(nullable=True)

    def test_polars_complex_types(self, reset_db: None) -> None:
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        # Test complex nested data types
        data = {
            'id': [1, 2],
            'struct_col': [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}],
            'list_of_lists': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            'mixed_list': [['a', 1, True], ['b', 2, False]]
        }
        
        # Create DataFrame with proper Polars types
        df = pl.DataFrame({
            'id': data['id'],
            'struct_col': data['struct_col'],  # This will be Object type in Polars
            'list_of_lists': data['list_of_lists'],
            'mixed_list': data['mixed_list']
        })
        
        t = pxt.io.import_polars('test_complex', df)
        
        schema = t._get_schema()
        assert schema['id'] == ts.IntType(nullable=True)
        # Complex types should map to JSON
        assert schema['struct_col'] == ts.JsonType(nullable=True)
        assert schema['list_of_lists'] == ts.JsonType(nullable=True)
        assert schema['mixed_list'] == ts.JsonType(nullable=True)
        
        # Verify data integrity
        res = t.select().order_by(t.id).collect()
        assert res['struct_col'] == data['struct_col']
        assert res['list_of_lists'] == data['list_of_lists']
        assert res['mixed_list'] == data['mixed_list']

    def test_polars_errors(self, reset_db: None) -> None:
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        data = {'id': [1, 2], 'name': ['Alice', None]}
        df = pl.DataFrame(data)
        
        # Test schema override with non-existent column
        with pytest.raises(excs.Error) as exc_info:
            pxt.io.import_polars(
                'test_errors', 
                df, 
                schema_overrides={'nonexistent_col': pxt.String}
            )
        assert 'Some column(s) specified in `schema_overrides` are not present' in str(exc_info.value)
        
        # Test primary key with null values
        with pytest.raises(excs.Error) as exc_info:
            pxt.io.import_polars('test_errors', df, primary_key='name')
        assert 'Primary key column `name` cannot contain null values.' in str(exc_info.value)

    def test_polars_not_available_error(self, reset_db: None, monkeypatch) -> None:
        # Mock Polars as not available
        monkeypatch.setattr('pixeltable.io.polars.POLARS_AVAILABLE', False)
        
        # This should raise an error about Polars not being available
        with pytest.raises(excs.Error) as exc_info:
            from pixeltable.io.polars import import_polars
            import_polars('test', None)  # DataFrame doesn't matter since we'll error first
        
        assert "Polars integration requires the 'polars' package" in str(exc_info.value)

    def test_categorical_and_special_types(self, reset_db: None) -> None:
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        # Test categorical and other special Polars types
        df = pl.DataFrame({
            'id': [1, 2, 3],
            'category': ['A', 'B', 'A'],
            'binary_data': [b'bytes1', b'bytes2', b'bytes3']
        }).with_columns([
            pl.col('category').cast(pl.Categorical),
            # Binary stays as is for now
        ])
        
        t = pxt.io.import_polars('test_special', df)
        
        schema = t._get_schema()
        assert schema['id'] == ts.IntType(nullable=True)
        assert schema['category'] == ts.StringType(nullable=True)  # Categorical -> String
        assert schema['binary_data'] == ts.StringType(nullable=True)  # Binary -> String
        
        # Verify data was converted properly
        res = t.select().order_by(t.id).collect()
        assert res['category'] == ['A', 'B', 'A'] 