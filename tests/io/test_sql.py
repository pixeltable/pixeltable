import datetime
import json
import pathlib
import uuid

import pytest
import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.io.sql import export_sql


class TestSql:
    """
    TODO:
    - test_export_snowflake(), analogous to test_export_sqlite()
    """

    def create_test_data(self, num_rows: int = 10_000) -> tuple[pxt.Table, list[dict]]:
        """Create test table with all exportable column types."""
        t = pxt.create_table(
            'test1',
            {
                'c_int': pxt.Int,
                'c_string': pxt.String,
                'c_float': pxt.Float,
                'c_bool': pxt.Bool,
                'c_timestamp': pxt.Timestamp,
                'c_date': pxt.Date,
                'c_uuid': pxt.UUID,
                'c_binary': pxt.Binary,
                'c_json': pxt.Json,
            },
        )
        base_date = datetime.date(2024, 1, 1)
        rows = [
            {
                'c_int': i,
                'c_string': f'row_{i}',
                'c_float': i * 1.5,
                'c_bool': i % 2 == 0,
                'c_timestamp': datetime.datetime.now() - datetime.timedelta(seconds=i),
                'c_date': base_date + datetime.timedelta(days=i),
                'c_uuid': uuid.uuid5(uuid.NAMESPACE_DNS, f'row_{i}'),
                'c_binary': f'binary_data_{i}'.encode(),
                'c_json': {'int_field': i, 'str_field': f'val_{i}', 'nested': {'data': i * 2}},
            }
            for i in range(num_rows)
        ]
        t.insert(rows)
        return t, rows

    def validate_schema(self, engine: sql.Engine, table_name: str, expected_columns: dict[str, type]) -> None:
        inspector = sql.inspect(engine)
        columns = {col['name']: col['type'] for col in inspector.get_columns(table_name)}
        assert set(columns.keys()) == set(expected_columns.keys())
        for col_name, col_type in expected_columns.items():
            assert type(columns[col_name]) is col_type

    def test_export_sqlite(self, reset_db: None, tmp_path: pathlib.Path) -> None:
        t, rows = self.create_test_data(100_000)
        db_path = tmp_path / 'test.db'
        connection_string = f'sqlite:///{db_path}'
        engine = sql.create_engine(connection_string)

        # Export full table
        export_sql(t, 'test_table', db_connect_str=connection_string)
        self.validate_schema(
            engine,
            'test_table',
            {
                'c_int': sql.INTEGER,
                'c_string': sql.VARCHAR,
                'c_float': sql.FLOAT,
                'c_bool': sql.BOOLEAN,
                'c_timestamp': sql.TIMESTAMP,
                'c_date': sql.DATE,
                'c_uuid': sql.NUMERIC,  # sqlalchemy maps that back incorrectly when loading the metadata
                'c_binary': sql.BLOB,
                'c_json': sql.dialects.sqlite.JSON,
            },
        )

        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table ORDER BY c_int')).fetchall()
            assert len(result) == len(rows)
            for col_idx, col_name in [(0, 'c_int'), (1, 'c_string'), (2, 'c_float'), (3, 'c_bool'), (7, 'c_binary')]:
                assert all(row[col_idx] == rows[i][col_name] for i, row in enumerate(result)), col_name

            assert all(
                datetime.datetime.fromisoformat(row[4]) == rows[i]['c_timestamp'] for i, row in enumerate(result)
            )
            assert all(row[5] == rows[i]['c_date'].isoformat() for i, row in enumerate(result))
            assert all(row[6] == rows[i]['c_uuid'].hex for i, row in enumerate(result))
            assert all(json.loads(row[8]) == rows[i]['c_json'] for i, row in enumerate(result))

        # Export subset of columns
        export_sql(t.select(t.c_int, t.c_string), 'test_table', db_connect_str=connection_string, if_exists='replace')
        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table ORDER BY c_int')).fetchall()
            assert len(result) == len(rows)
            assert all(row[0] == rows[i]['c_int'] and row[1] == rows[i]['c_string'] for i, row in enumerate(result))
            for col_idx, col_name in enumerate(['c_int', 'c_string']):
                assert all(row[col_idx] == rows[i][col_name] for i, row in enumerate(result)), col_name

        # Export subset of rows
        export_sql(t.where(t.c_int < 10), 'test_table', db_connect_str=connection_string, if_exists='replace')
        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table ORDER BY c_int')).fetchall()
            assert len(result) == 10

    def test_export_postgresql(self, reset_db: None) -> None:
        t, rows = self.create_test_data(100_000)
        connection_string = Env.get().db_url
        engine = sql.create_engine(connection_string)

        # Export full table
        export_sql(t, 'test_table', db_connect_str=connection_string)
        self.validate_schema(
            engine,
            'test_table',
            {
                'c_int': sql.INTEGER,
                'c_string': sql.VARCHAR,
                'c_float': sql.dialects.postgresql.DOUBLE_PRECISION,
                'c_bool': sql.BOOLEAN,
                'c_timestamp': sql.dialects.postgresql.TIMESTAMP,
                'c_date': sql.DATE,
                'c_uuid': sql.UUID,
                'c_binary': sql.dialects.postgresql.BYTEA,
                'c_json': sql.dialects.postgresql.JSONB,
            },
        )

        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table ORDER BY c_int')).fetchall()
            assert len(result) == len(rows)
            for col_idx, col_name in enumerate(
                ['c_int', 'c_string', 'c_float', 'c_bool', 'c_timestamp', 'c_date', 'c_uuid', 'c_binary', 'c_json']
            ):
                assert all(row[col_idx] == rows[i][col_name] for i, row in enumerate(result)), col_name
            # assert bytes(row[7]) == rows[i]['c_binary']

        # insert into the same table
        export_sql(t, 'test_table', db_connect_str=connection_string, if_exists='insert')
        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table ORDER BY c_int')).fetchall()
            assert len(result) == 2 * len(rows)

        # Export subset of columns
        export_sql(t.select(t.c_int, t.c_string), 'test_table', db_connect_str=connection_string, if_exists='replace')
        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table ORDER BY c_int')).fetchall()
            assert len(result) == len(rows)
            for col_idx, col_name in enumerate(['c_int', 'c_string']):
                assert all(row[col_idx] == rows[i][col_name] for i, row in enumerate(result)), col_name

        # Export subset of rows
        export_sql(t.where(t.c_int < 10), 'test_table', db_connect_str=connection_string, if_exists='replace')
        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table ORDER BY c_int')).fetchall()
            assert len(result) == 10

    def test_errors(self, reset_db: None) -> None:
        connection_string = Env.get().db_url

        # unsupported column type
        t_img = pxt.create_table('test_img', {'img': pxt.Image})
        with pytest.raises(pxt.Error, match='Cannot export column of type'):
            export_sql(t_img, 'error_table', db_connect_str=connection_string)

        # table exists with if_exists='error'
        t, _ = self.create_test_data(10)
        export_sql(t, 'existing_table', db_connect_str=connection_string)
        with pytest.raises(pxt.Error, match='already exists'):
            export_sql(t, 'existing_table', db_connect_str=connection_string, if_exists='error')

        # missing column in target table
        t2 = pxt.create_table('test2', {'c_int': pxt.Int, 'c_string': pxt.String, 'extra': pxt.Int})
        t2.insert([{'c_int': 1, 'c_string': 'a', 'extra': 100}])
        with pytest.raises(pxt.Error, match="Column 'extra' not in table"):
            export_sql(t2, 'existing_table', db_connect_str=connection_string, if_exists='insert')

        # incompatible schema
        t3 = pxt.create_table('test3', {'c_int': pxt.Json})
        t3.insert([{'c_int': {'key': 'value'}}])
        with pytest.raises(pxt.Error, match=r"column 'c_int' of type INTEGER is not compatible"):
            export_sql(t3, 'existing_table', db_connect_str=connection_string, if_exists='insert')
