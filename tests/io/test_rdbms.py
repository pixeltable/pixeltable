import datetime
import pathlib

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.io.rdbms import export_rdbms


class TestRdbms:
    def test_export_rdbms(self, reset_db: None, tmp_path: pathlib.Path) -> None:
        from zoneinfo import ZoneInfo

        # Note: JSON columns are not tested here because export_rdbms converts them to JSONB,
        # which SQLite doesn't support. JSON export should be tested with PostgreSQL.
        t = pxt.create_table('test1', {'c1': pxt.Int, 'c2': pxt.String, 'c3': pxt.Timestamp})

        tz = ZoneInfo('America/Anchorage')
        ts1 = datetime.datetime(2012, 1, 1, 12, 0, 0, 25, tz)
        ts2 = datetime.datetime(2012, 2, 1, 12, 0, 0, 25, tz)
        t.insert([{'c1': 1, 'c2': 'row1', 'c3': ts1}, {'c1': 2, 'c2': 'row2', 'c3': ts2}])

        db_path = tmp_path / 'test.db'
        connection_string = f'sqlite:///{db_path}'

        # Export full table
        export_rdbms(t, 'test_table1', connection_string=connection_string)

        # Verify export
        engine = sql.create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table1 ORDER BY c1')).fetchall()
            assert len(result) == 2
            assert result[0][0] == 1
            assert result[0][1] == 'row1'
            assert result[1][0] == 2
            assert result[1][1] == 'row2'

        # Export with select() - subset of columns
        export_rdbms(t.select(t.c1, t.c2), 'test_table2', connection_string=connection_string)

        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table2 ORDER BY c1')).fetchall()
            assert len(result) == 2
            assert len(result[0]) == 2  # Only 2 columns
            assert result[0][0] == 1
            assert result[0][1] == 'row1'
            assert result[1][0] == 2
            assert result[1][1] == 'row2'

        # Export with where() - filtered rows
        export_rdbms(t.where(t.c1 == 1), 'test_table3', connection_string=connection_string)

        with engine.connect() as conn:
            result = conn.execute(sql.text('SELECT * FROM test_table3 ORDER BY c1')).fetchall()
            assert len(result) == 1
            assert result[0][0] == 1
            assert result[0][1] == 'row1'
