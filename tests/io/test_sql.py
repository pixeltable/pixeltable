import datetime
import json
import pathlib

import pytest
import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.io.sql import export_sql


class TestSql:
    def create_test_data(self, num_rows: int = 10_000) -> tuple[pxt.Table, list[dict]]:
        t = pxt.create_table('test1', {'c1': pxt.Int, 'c2': pxt.String, 'c3': pxt.Timestamp, 'c4': pxt.Json})
        rows = [
            {
                'c1': i,
                'c2': f'row_{i}',
                'c3': datetime.datetime.now() - datetime.timedelta(seconds=i),
                'c4': {'int_field': i, 'str_field': f'val_{i}', 'nested': {'data': i * 2}},
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
        t, rows = self.create_test_data(10_000)
        db_path = tmp_path / 'test.db'
        connection_string = f'sqlite:///{db_path}'

        # Export full table
        export_sql(t, 'test_table1', connection_string=connection_string)

        # Verify export
        engine = sql.create_engine(connection_string)

        self.validate_schema(
            engine,
            'test_table1',
            {'c1': sql.INTEGER, 'c2': sql.VARCHAR, 'c3': sql.TIMESTAMP, 'c4': sql.dialects.sqlite.json.JSON},
        )

        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table1 ORDER BY c1')).fetchall()
            assert len(result) == len(rows)
            for col_idx, col_name in enumerate(['c1', 'c2']):
                assert all(row[col_idx] == rows[i][col_name] for i, row in enumerate(result))
            # timestamps and json are returned as strings
            assert all(datetime.datetime.fromisoformat(row[2]) == rows[i]['c3'] for i, row in enumerate(result))
            assert all(json.loads(row[3]) == rows[i]['c4'] for i, row in enumerate(result))

        # Export subset of columns
        export_sql(t.select(t.c1, t.c2), 'test_table2', connection_string=connection_string)
        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table2 ORDER BY c1')).fetchall()
            assert len(result) == len(rows)
            for col_idx, col_name in enumerate(['c1', 'c2']):
                assert all(row[col_idx] == rows[i][col_name] for i, row in enumerate(result))

        # Export subset of rows
        export_sql(t.where(t.c1 < 100), 'test_table3', connection_string=connection_string)
        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table3 ORDER BY c1')).fetchall()
            assert len(result) == 100

    def test_export_postgresql(self, reset_db: None) -> None:
        t, rows = self.create_test_data()
        connection_string = Env.get().db_url

        # Export full table
        export_sql(t, 'test_export_table1', connection_string=connection_string)

        engine = sql.create_engine(connection_string)

        self.validate_schema(
            engine,
            'test_export_table1',
            {
                'c1': sql.INTEGER,
                'c2': sql.VARCHAR,
                'c3': sql.dialects.postgresql.types.TIMESTAMP,
                'c4': sql.dialects.postgresql.json.JSONB,
            },
        )

        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_export_table1 ORDER BY c1')).fetchall()
            assert len(result) == len(rows)
            for col_idx, col_name in enumerate(['c1', 'c2', 'c3', 'c4']):
                assert all(row[col_idx] == rows[i][col_name] for i, row in enumerate(result))

        export_sql(t.select(t.c1, t.c2), 'test_export_table2', connection_string=connection_string)
        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_export_table2 ORDER BY c1')).fetchall()
            assert len(result) == len(rows)
            for col_idx, col_name in enumerate(['c1', 'c2']):
                assert all(row[col_idx] == rows[i][col_name] for i, row in enumerate(result))

        export_sql(t.where(t.c1 < 100), 'test_export_table3', connection_string=connection_string)
        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_export_table3 ORDER BY c1')).fetchall()
            assert len(result) == 100
            for col_idx, col_name in enumerate(['c1', 'c2', 'c3', 'c4']):
                assert all(row[col_idx] == rows[i][col_name] for i, row in enumerate(result))

   edef test_errors(self, reset_db: None) -> None:
        connection_string = Env.get().db_url

        # 1. Unsupported column type (Image)
        t_img = pxt.create_table('test_img', {'img': pxt.Image})
        with pytest.raises(pxt.Error, match='Cannot export column of type'):
            export_sql(t_img, 'error_table', connection_string=connection_string)

        # 2. Table exists with if_exists='error'
        t, _ = self.create_test_data(10)
        export_sql(t, 'existing_table', connection_string=connection_string)
        with pytest.raises(pxt.Error, match='already exists'):
            export_sql(t, 'existing_table', connection_string=connection_string, if_exists='error')

        # 3. Missing column in target table
        t2 = pxt.create_table('test2', {'c1': pxt.Int, 'c2': pxt.String, 'extra': pxt.Int})
        t2.insert([{'c1': 1, 'c2': 'a', 'extra': 100}])
        with pytest.raises(pxt.Error, match='not in table'):
            export_sql(t2, 'existing_table', connection_string=connection_string, if_exists='append')
