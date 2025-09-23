import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.env import Env

from ..utils import skip_test_if_not_installed

pytest_plugins: list[str] = []


class TestPolars:
    def make_src_data(self) -> dict[str, Any]:
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
            'array_col_0': [np.array([1, 2]), np.array([3, 4])],  # numpy array column
            'json_col_1': [[1, 2], [3, 4]],
            'json_col_2': [{'a': 1}, {'b': 2}],
        }
        return src_data

    def test_polars_types_create(self, reset_db: None) -> None:
        skip_test_if_not_installed('polars')
        import polars as pl

        default_tz = Env.get().default_time_zone

        src_data = self.make_src_data()
        df = pl.DataFrame(src_data)
        t = pxt.create_table('test_types', source=df)

        # Verify the schema matches expected types
        expected_schema = {
            'int_col': ts.IntType(nullable=True),
            'float_col': ts.FloatType(nullable=True),
            'bool_col': ts.BoolType(nullable=True),
            'str_col': ts.StringType(nullable=True),
            'dt_col': ts.TimestampType(nullable=True),
            'aware_dt_col': ts.TimestampType(nullable=True),
            'date_col': ts.DateType(nullable=True),
            'array_col_0': ts.ArrayType(shape=(2,), dtype=ts.IntType(), nullable=True),
            'json_col_1': ts.JsonType(nullable=True),
            'json_col_2': ts.JsonType(nullable=True),  # Struct becomes JSON
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
        assert res['json_col_1'] == [[1, 2], [3, 4]]
        # polars infers struct schema from first record, so only 'a' field is preserved
        assert res['json_col_2'] == [{'a': 1}, {'a': None}]
        assert t.count() == len(df)

    def test_polars_types_insert(self, reset_db: None) -> None:
        skip_test_if_not_installed('polars')
        import polars as pl

        src_data = self.make_src_data()
        df = pl.DataFrame(src_data)
        t = pxt.create_table('test_types', source=df)

        initial_count = t.count()
        assert initial_count == len(df)

        # Insert the same DataFrame again
        t.insert(df)
        assert t.count() == 2 * initial_count

    def test_polars_with_schema_overrides(self, reset_db: None) -> None:
        skip_test_if_not_installed('polars')
        import polars as pl

        # Create a DataFrame with a string column that we want to override as Image
        # Use actual local image files that exist in the test data
        data: dict[str, Any] = {
            'name': ['image1', 'image2'],
            'url': ['docs/resources/images/000000000001.jpg', 'docs/resources/images/000000000009.jpg'],
            'score': [0.95, 0.87],
        }
        df = pl.DataFrame(data)

        # Override the url column to be Image type
        t = pxt.create_table('test_overrides', source=df, schema_overrides={'url': pxt.Image})

        schema = t._get_schema()
        assert schema['name'] == ts.StringType(nullable=True)
        assert schema['url'] == ts.ImageType(nullable=True)
        assert schema['score'] == ts.FloatType(nullable=True)

    def test_polars_with_primary_key(self, reset_db: None) -> None:
        skip_test_if_not_installed('polars')
        import polars as pl

        data: dict[str, Any] = {
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
        }
        df = pl.DataFrame(data)

        t = pxt.create_table('test_pk', source=df, primary_key='id')

        schema = t._get_schema()
        # Primary key columns should be non-nullable
        assert schema['id'] == ts.IntType(nullable=False)
        assert schema['name'] == ts.StringType(nullable=True)
        assert schema['email'] == ts.StringType(nullable=True)

    def test_polars_complex_types(self, reset_db: None) -> None:
        skip_test_if_not_installed('polars')
        import polars as pl

        # Test complex nested data types
        data: dict[str, Any] = {
            'id': [1, 2],
            'struct_col': [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}],
            'list_of_lists': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            'mixed_list': [['a', 1, True], ['b', 2, False]],
        }

        # Create DataFrame with proper polars types
        df = pl.DataFrame(
            {
                'id': data['id'],
                'struct_col': data['struct_col'],  # This will be Object type in polars
                'list_of_lists': data['list_of_lists'],
                'mixed_list': data['mixed_list'],
            },
            strict=False,  # Allow mixed types in lists
        )

        t = pxt.create_table('test_complex', source=df)

        schema = t._get_schema()
        assert schema['id'] == ts.IntType(nullable=True)
        # Complex types should map to JSON
        assert schema['struct_col'] == ts.JsonType(nullable=True)
        assert schema['list_of_lists'] == ts.JsonType(nullable=True)
        # mixed_list becomes List(String) in polars, so it's an ArrayType
        assert schema['mixed_list'] == ts.JsonType(nullable=True)  # List of strings with mixed lengths - use JSON

        # Verify data integrity
        res = t.select().order_by(t.id).collect()
        assert res['struct_col'] == data['struct_col']
        assert res['list_of_lists'] == data['list_of_lists']
        # polars converts mixed types to strings in lists
        assert res['mixed_list'] == [['a', '1', 'true'], ['b', '2', 'false']]

    def test_polars_errors(self, reset_db: None) -> None:
        skip_test_if_not_installed('polars')
        import polars as pl

        data: dict[str, Any] = {'id': [1, 2], 'name': ['Alice', None]}
        df = pl.DataFrame(data)

        # Test schema override with non-existent column
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test_errors', source=df, schema_overrides={'nonexistent_col': pxt.String})
        assert 'Some column(s) specified in `schema_overrides` are not present' in str(exc_info.value)

        # Test primary key with null values
        with pytest.raises(excs.Error, match='cannot contain null values'):
            pxt.create_table('test_errors', source=df, primary_key='name')

        data: dict[str, Any] = {'id': [1, 2, 3], 'binary': [b'bytes1', b'bytes2\x11\x02', b'bytes3']}
        df = pl.DataFrame(data)

        with pytest.raises(pxt.Error, match='Could not infer Pixeltable type for polars column'):
            pxt.create_table('test_errors', source=df)

    def test_polars_categorical_and_special_types(self, reset_db: None) -> None:
        # Test categorical and other special polars types
        skip_test_if_not_installed('polars')
        import polars as pl

        df = pl.DataFrame({'id': [1, 2, 3], 'category': ['A', 'B', 'A']}).with_columns(
            [pl.col('category').cast(pl.Categorical)]
        )
        print(df)
        t = pxt.create_table('test_special', source=df)

        schema = t._get_schema()
        assert schema['id'] == ts.IntType(nullable=True)
        assert schema['category'] == ts.StringType(nullable=True)  # Categorical -> String

        # Verify data was converted properly
        res = t.select().order_by(t.id).collect()
        assert res['category'] == ['A', 'B', 'A']

    def test_polars_comprehensive_polars_integration(self, reset_db: None) -> None:
        """Comprehensive test of all polars types and complex functionality using create_table(source=df)"""
        skip_test_if_not_installed('polars')
        import polars as pl

        # Create the most comprehensive polars DataFrame possible
        complex_data: dict[str, Any] = {
            # Basic types
            'int8_col': pl.Series([1, 2, 3], dtype=pl.Int8),
            'int16_col': pl.Series([100, 200, 300], dtype=pl.Int16),
            'int32_col': pl.Series([1000, 2000, 3000], dtype=pl.Int32),
            'int64_col': pl.Series([10000, 20000, 30000], dtype=pl.Int64),
            'uint8_col': pl.Series([1, 2, 3], dtype=pl.UInt8),
            'uint16_col': pl.Series([100, 200, 300], dtype=pl.UInt16),
            'uint32_col': pl.Series([1000, 2000, 3000], dtype=pl.UInt32),
            'uint64_col': pl.Series([10000, 20000, 30000], dtype=pl.UInt64),
            'float32_col': pl.Series([1.1, 2.2, 3.3], dtype=pl.Float32),
            'float64_col': pl.Series([1.11, 2.22, 3.33], dtype=pl.Float64),
            'bool_col': [True, False, True],
            'string_col': ['hello', 'world', 'polars'],
            # Date and time types
            'date_col': [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)],
            'datetime_col': [
                datetime.datetime(2024, 1, 1, 12, 0, 0),
                datetime.datetime(2024, 1, 2, 13, 0, 0),
                datetime.datetime(2024, 1, 3, 14, 0, 0),
            ],
            'datetime_tz_col': [
                datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
                datetime.datetime(2024, 1, 2, 13, 0, 0, tzinfo=datetime.timezone.utc),
                datetime.datetime(2024, 1, 3, 14, 0, 0, tzinfo=datetime.timezone.utc),
            ],
            # Complex nested types
            'list_int_col': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'list_float_col': [[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]],
            'list_string_col': [['a', 'b'], ['c', 'd'], ['e', 'f']],
            'list_mixed_length': [[1, 2], [3, 4, 5], [6]],
            # Struct (nested object) types
            'struct_col': [
                {'name': 'Alice', 'age': 30, 'active': True},
                {'name': 'Bob', 'age': 25, 'active': False},
                {'name': 'Charlie', 'age': 35, 'active': True},
            ],
            # Deeply nested structures
            'nested_complex': [
                {'user': {'name': 'Alice', 'scores': [90, 85, 88]}, 'meta': {'region': 'US'}},
                {'user': {'name': 'Bob', 'scores': [75, 80, 82]}, 'meta': {'region': 'EU'}},
                {'user': {'name': 'Charlie', 'scores': [95, 92, 89]}, 'meta': {'region': 'ASIA'}},
            ],
            # Large arrays for testing performance
            'large_array_col': [
                np.array(list(range(100))),
                np.array(list(range(100, 200))),
                np.array(list(range(200, 300))),
            ],
            # Null handling
            'nullable_int': [1, None, 3],
            'nullable_string': ['test', None, 'value'],
            'nullable_list': [[1, 2], None, [3, 4]],
        }

        # Create polars DataFrame with various casting operations
        df = pl.DataFrame(complex_data)

        # Add categorical column properly
        categorical_data = pl.DataFrame({'categorical_col': ['category_A', 'category_B', 'category_A']})
        categorical_data = categorical_data.with_columns([pl.col('categorical_col').cast(pl.Categorical)])
        df = df.with_columns([categorical_data['categorical_col']])

        # Test 1: Create table with source parameter
        table = pxt.create_table('comprehensive_polars_test', source=df)

        # Verify row count
        assert table.count() == 3

        # Test 2: Verify schema inference for all types
        schema = table._get_schema()

        # Basic integer types should all map to IntType
        for col in [
            'int8_col',
            'int16_col',
            'int32_col',
            'int64_col',
            'uint8_col',
            'uint16_col',
            'uint32_col',
            'uint64_col',
        ]:
            assert schema[col] == ts.IntType(nullable=True), f'Failed for {col}'

        # Float types should map to FloatType
        for col in ['float32_col', 'float64_col']:
            assert schema[col] == ts.FloatType(nullable=True), f'Failed for {col}'

        # Boolean and string types
        assert schema['bool_col'] == ts.BoolType(nullable=True)
        assert schema['string_col'] == ts.StringType(nullable=True)

        # Date and time types
        assert schema['date_col'] == ts.DateType(nullable=True)
        assert schema['datetime_col'] == ts.TimestampType(nullable=True)
        assert schema['datetime_tz_col'] == ts.TimestampType(nullable=True)

        # Complex types should map to appropriate Pixeltable types
        # Lists of integers with consistent length should become ArrayType
        #        assert schema['list_int_col'] == ts.ArrayType(shape=(None, 3), dtype=ts.IntType(), nullable=True)

        # Lists of mixed length - shape inferred from first element
        #        assert schema['list_mixed_length'] == ts.ArrayType(shape=(None, 2), dtype=ts.IntType(), nullable=True)

        # Structs should become JsonType
        assert schema['struct_col'] == ts.JsonType(nullable=True)
        assert schema['nested_complex'] == ts.JsonType(nullable=True)

        # Categorical should become StringType
        assert schema['categorical_col'] == ts.StringType(nullable=True)

        # Large arrays should be handled
        assert schema['large_array_col'] == ts.ArrayType(shape=(100,), dtype=ts.IntType(), nullable=True)

        # Test 3: Verify data integrity by retrieving and checking values
        results = table.select().order_by(table.int32_col).collect()

        # Check basic type conversions
        assert results['int32_col'] == [1000, 2000, 3000]
        assert results['float64_col'] == [1.11, 2.22, 3.33]
        assert results['bool_col'] == [True, False, True]
        assert results['string_col'] == ['hello', 'world', 'polars']

        # Check date/time handling
        assert len(results['date_col']) == 3
        assert all(isinstance(d, datetime.date) for d in results['date_col'])

        # Check complex data integrity
        assert results['struct_col'][0] == {'name': 'Alice', 'age': 30, 'active': True}
        assert results['list_int_col'][0] == [1, 2, 3]  # Should be numpy array

        # Check null handling
        assert results['nullable_int'] == [1, None, 3]
        assert results['nullable_string'] == ['test', None, 'value']

        # Test 4: Create table with schema overrides
        pxt.drop_table('comprehensive_polars_override', if_not_exists='ignore')
        override_table = pxt.create_table(
            'comprehensive_polars_override',
            source=df,
            schema_overrides={
                'string_col': pxt.String,  # Should remain string
                'large_array_col': pxt.Json,  # Override array to JSON
            },
        )

        override_schema = override_table._get_schema()
        assert override_schema['string_col'] == ts.StringType(nullable=True)
        assert override_schema['large_array_col'] == ts.JsonType(nullable=True)

        # Test 5: Create table with primary key from complex data
        pxt.drop_table('comprehensive_polars_pk', if_not_exists='ignore')
        pk_table = pxt.create_table(
            'comprehensive_polars_pk',
            source=df,
            primary_key='int32_col',  # Use int32_col as primary key
        )

        pk_schema = pk_table._get_schema()
        # Primary key column should be non-nullable
        assert pk_schema['int32_col'] == ts.IntType(nullable=False)
        # Other columns should remain nullable
        assert pk_schema['string_col'] == ts.StringType(nullable=True)

        # Test 6: Insert additional data into the table
        additional_data: dict[str, Any] = {
            'int8_col': pl.Series([4, 5], dtype=pl.Int8),
            'int16_col': pl.Series([400, 500], dtype=pl.Int16),
            'int32_col': pl.Series([4000, 5000], dtype=pl.Int32),
            'int64_col': pl.Series([40000, 50000], dtype=pl.Int64),
            'uint8_col': pl.Series([4, 5], dtype=pl.UInt8),
            'uint16_col': pl.Series([400, 500], dtype=pl.UInt16),
            'uint32_col': pl.Series([4000, 5000], dtype=pl.UInt32),
            'uint64_col': pl.Series([40000, 50000], dtype=pl.UInt64),
            'float32_col': pl.Series([4.4, 5.5], dtype=pl.Float32),
            'float64_col': pl.Series([4.44, 5.55], dtype=pl.Float64),
            'bool_col': [False, True],
            'string_col': ['additional', 'data'],
            'date_col': [datetime.date(2024, 1, 4), datetime.date(2024, 1, 5)],
            'datetime_col': [datetime.datetime(2024, 1, 4, 15, 0, 0), datetime.datetime(2024, 1, 5, 16, 0, 0)],
            'datetime_tz_col': [
                datetime.datetime(2024, 1, 4, 15, 0, 0, tzinfo=datetime.timezone.utc),
                datetime.datetime(2024, 1, 5, 16, 0, 0, tzinfo=datetime.timezone.utc),
            ],
            'list_int_col': [[10, 11, 12], [13, 14, 15]],
            'list_float_col': [[7.7, 8.8], [9.9, 10.0]],
            'list_string_col': [['g', 'h'], ['i', 'j']],
            'list_mixed_length': [[7, 8, 9, 10], [11]],
            'struct_col': [{'name': 'David', 'age': 40, 'active': True}, {'name': 'Eve', 'age': 28, 'active': False}],
            'nested_complex': [
                {'user': {'name': 'David', 'scores': [88, 90, 92]}, 'meta': {'region': 'CA'}},
                {'user': {'name': 'Eve', 'scores': [78, 85, 88]}, 'meta': {'region': 'AU'}},
            ],
            'large_array_col': [np.array(range(300, 400)), np.array(range(400, 500))],
            'nullable_int': [4, 5],
            'nullable_string': ['more', 'test'],
            'nullable_list': [[5, 6], [7, 8]],
            'categorical_col': ['category_C', 'category_A'],
        }

        additional_df = pl.DataFrame(additional_data)
        additional_df = additional_df.with_columns([pl.col('categorical_col').cast(pl.Categorical)])

        # Insert using polars DataFrame
        table.insert(additional_df)
        assert table.count() == 5  # 3 original + 2 additional

        # Verify the additional data was inserted correctly
        all_results = table.select().order_by(table.int32_col).collect()
        assert len(all_results) == 5
        assert all_results['string_col'][-2:] == ['additional', 'data']

        # Test 7: Verify to_polars() method on DataFrameResultSet
        print('Test 7: Testing to_polars() conversion...')
        result_set = table.select().order_by(table.int32_col).collect()

        # Convert back to polars DataFrame
        converted_df = result_set.to_polars()

        # Verify the conversion preserved data integrity
        assert len(converted_df) == 5  # Should have all 5 rows
        assert converted_df.width == len(schema)  # Should have all columns
        assert converted_df.columns == list(schema.keys())  # Column names should match

        # Verify some key data points
        assert converted_df['int32_col'].to_list() == [1000, 2000, 3000, 4000, 5000]
        assert converted_df['string_col'].to_list() == ['hello', 'world', 'polars', 'additional', 'data']

        # Test empty result conversion
        empty_result = table.select().where(table.int32_col > 10000).collect()
        empty_df = empty_result.to_polars()
        assert len(empty_df) == 0
        assert empty_df.columns == list(schema.keys())  # Should preserve column structure
        print('✅ to_polars() conversion working perfectly')

        print(f'✅ Comprehensive polars integration test passed with {table.count()} rows and {len(schema)} columns!')
