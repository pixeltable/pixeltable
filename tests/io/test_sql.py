import dataclasses
import datetime
import json
import pathlib
import uuid
from typing import Any, Callable

import sqlalchemy as sql
import sqlalchemy.dialects.postgresql
import sqlalchemy.dialects.sqlite  # noqa: F401

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.io.sql import export_sql

from ..utils import pxt_raises


@dataclasses.dataclass(frozen=True)
class _DialectSpec:
    name: str
    connect: Callable[[pathlib.Path], str]  # tmp_path -> connection string
    sa_types: dict[str, type]  # pxt col -> SA type for create-table assertions
    decode: dict[str, Callable[[Any], Any]]  # SA value -> python value comparable to rows[i][col]
    # if_exists='insert' requires reflecting the target schema, which roundtrips through SA types;
    # sqlite's UUID -> NUMERIC mismatch makes the compat probe fail, so skip it there.
    supports_insert: bool = True


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

    def _sqlite_spec(self) -> _DialectSpec:
        return _DialectSpec(
            name='sqlite',
            connect=lambda tmp: f'sqlite:///{tmp / "test.db"}',
            sa_types={
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
            decode={
                'c_timestamp': datetime.datetime.fromisoformat,
                'c_date': datetime.date.fromisoformat,
                'c_uuid': lambda v: uuid.UUID(hex=v),
                'c_json': json.loads,
            },
            supports_insert=False,
        )

    def _postgresql_spec(self) -> _DialectSpec:
        return _DialectSpec(
            name='postgresql',
            connect=lambda _: Env.get().db_url,
            sa_types={
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
            decode={},
        )

    def _verify_export(
        self,
        engine: sql.Engine,
        table_name: str,
        spec: _DialectSpec,
        rows: list[dict],
        col_names: list[str] | None = None,
    ) -> None:
        """Schema + value verification. col_names defaults to all spec.sa_types keys."""
        cols = col_names if col_names is not None else list(spec.sa_types.keys())
        expected = {c: spec.sa_types[c] for c in cols}
        inspector = sql.inspect(engine)
        actual = {col['name']: col['type'] for col in inspector.get_columns(table_name)}
        assert set(actual.keys()) == set(cols)
        assert all(type(actual[c]) is expected[c] for c in cols)

        select = ', '.join(cols)
        with engine.connect() as conn:
            result = conn.execute(sql.text(f'SELECT {select} FROM {table_name} ORDER BY c_int')).fetchall()
        assert len(result) == len(rows)
        identity = lambda v: v  # noqa: E731
        for i, row in enumerate(result):
            for j, c in enumerate(cols):
                decoded = spec.decode.get(c, identity)(row[j])
                assert decoded == rows[i][c], (c, i, decoded, rows[i][c])

    def _row_count(self, engine: sql.Engine, table_name: str) -> int:
        with engine.connect() as conn:
            return conn.execute(sql.text(f'SELECT COUNT(*) FROM {table_name}')).scalar_one()

    def _run_export_suite(self, spec: _DialectSpec, tmp_path: pathlib.Path) -> None:
        t, rows = self.create_test_data(100_000)
        connect = spec.connect(tmp_path)
        engine = sql.create_engine(connect)

        # full export (default if_not_exists='create')
        export_sql(t, 'test_table', db_connect_str=connect)
        self._verify_export(engine, 'test_table', spec, rows)

        # if_exists='insert' appends rows
        if spec.supports_insert:
            export_sql(t, 'test_table', db_connect_str=connect, if_exists='insert')
            assert self._row_count(engine, 'test_table') == 2 * len(rows)

        # if_exists='replace' + subset of columns
        export_sql(t.select(t.c_int, t.c_string), 'test_table', db_connect_str=connect, if_exists='replace')
        self._verify_export(engine, 'test_table', spec, rows, col_names=['c_int', 'c_string'])

        # if_exists='replace' + subset of rows
        export_sql(t.where(t.c_int < 10), 'test_table', db_connect_str=connect, if_exists='replace')
        assert self._row_count(engine, 'test_table') == 10

        # if_not_exists='error' against missing target
        with pxt_raises(pxt.ErrorCode.PATH_NOT_FOUND, match=r"table 'never' does not exist"):
            export_sql(t, 'never', db_connect_str=connect, if_not_exists='error')

        # if_not_exists='create' explicit (creates a fresh table)
        export_sql(t.where(t.c_int < 5), 'fresh_table', db_connect_str=connect, if_not_exists='create')
        assert self._row_count(engine, 'fresh_table') == 5

    def test_export_sqlite(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        self._run_export_suite(self._sqlite_spec(), tmp_path)

    def test_export_postgresql(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        self._run_export_suite(self._postgresql_spec(), tmp_path)

    def test_errors(self, uses_db: None) -> None:
        connection_string = Env.get().db_url

        # unsupported column type
        t_img = pxt.create_table('test_img', {'img': pxt.Image})
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot export column of type'):
            export_sql(t_img, 'error_table', db_connect_str=connection_string)

        # table exists with if_exists='error'
        t, _ = self.create_test_data(10)
        export_sql(t, 'existing_table', db_connect_str=connection_string)
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists'):
            export_sql(t, 'existing_table', db_connect_str=connection_string, if_exists='error')

        # missing column in target table
        t2 = pxt.create_table('test2', {'c_int': pxt.Int, 'c_string': pxt.String, 'extra': pxt.Int})
        t2.insert([{'c_int': 1, 'c_string': 'a', 'extra': 100}])
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match="column 'extra' not in table"):
            export_sql(t2, 'existing_table', db_connect_str=connection_string, if_exists='insert')

        # incompatible schema
        t3 = pxt.create_table('test3', {'c_int': pxt.Json})
        t3.insert([{'c_int': {'key': 'value'}}])
        with pxt_raises(pxt.ErrorCode.TYPE_MISMATCH, match=r"column 'c_int' of type INTEGER"):
            export_sql(t3, 'existing_table', db_connect_str=connection_string, if_exists='insert')
