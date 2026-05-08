import dataclasses
import datetime
import json
import pathlib
import uuid
from typing import Any, Callable

import pytest
import sqlalchemy as sql
import sqlalchemy.dialects.postgresql
import sqlalchemy.dialects.sqlite  # noqa: F401

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.io.sql import export_sql, import_sql

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

        # non-scalar source type into an existing target column
        t_str = pxt.create_table('img_target_seed', {'img': pxt.String})
        t_str.insert([{'img': 'placeholder'}])
        export_sql(t_str, 'img_target', db_connect_str=connection_string)
        t_img2 = pxt.create_table('test_img2', {'img': pxt.Image})
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r"column 'img' of source type Image"):
            export_sql(t_img2, 'img_target', db_connect_str=connection_string, if_exists='insert')


_IMPORT_DIALECTS = ['sqlite', 'postgresql']


def _import_engine(dialect: str, tmp_path: pathlib.Path) -> sql.Engine:
    """Build a SQLAlchemy Engine for use as an `import_sql` source. The Postgres dialect points at pixeltable's
    embedded database, which `uses_db` resets before each test (see `clean_db`)."""
    if dialect == 'sqlite':
        return sql.create_engine(f'sqlite:///{tmp_path / "import_src.db"}')
    if dialect == 'postgresql':
        return sql.create_engine(Env.get().db_url)
    raise AssertionError(dialect)


def _seed_source(engine: sql.Engine, table_name: str, columns: list[sql.Column], rows: list[dict]) -> sql.Table:
    """Create `table_name` with `columns` in `engine` and insert `rows`. Returns the SA Table."""
    meta = sql.MetaData()
    src = sql.Table(table_name, meta, *columns)
    meta.create_all(engine)
    if rows:
        with engine.begin() as conn:
            conn.execute(src.insert(), rows)
    return src


class TestImportSql:
    """import_sql() tests, parametrized across SQLite (file) and the embedded Postgres."""

    @pytest.mark.parametrize('dialect', _IMPORT_DIALECTS)
    def test_import_full_table(self, uses_db: None, tmp_path: pathlib.Path, dialect: str) -> None:
        """End-to-end import of a full SA Table: type inference for all common SA types, nullable vs non-nullable
        propagation, NULL-to-None value roundtrip, exact value preservation, and the single-version-bump claim
        across batch boundaries (rows > BATCH_SIZE so streaming actually engages)."""
        engine = _import_engine(dialect, tmp_path)
        n = 2500  # > SqlSourceNode.BATCH_SIZE (1024) to force at least 3 batches

        # mix of nullable and non-nullable columns; nullable columns include NULL values
        src = _seed_source(
            engine,
            'src_full',
            [
                sql.Column('c_int', sql.Integer, nullable=False),
                sql.Column('c_str', sql.String, nullable=False),
                sql.Column('c_float', sql.Float, nullable=True),
                sql.Column('c_bool', sql.Boolean, nullable=True),
                sql.Column('c_ts', sql.DateTime, nullable=True),
                sql.Column('c_date', sql.Date, nullable=True),
                sql.Column('c_json', sql.JSON, nullable=True),
                sql.Column('c_bytes', sql.LargeBinary, nullable=True),
            ],
            [
                {
                    'c_int': i,
                    'c_str': f'row_{i}',
                    # every 7th row uses NULL so we exercise both NULL and value paths
                    'c_float': None if i % 7 == 0 else float(i) * 1.5,
                    'c_bool': None if i % 7 == 0 else (i % 2 == 0),
                    'c_ts': None if i % 7 == 0 else datetime.datetime(2024, 1, 1) + datetime.timedelta(seconds=i),
                    'c_date': None if i % 7 == 0 else datetime.date(2024, 1, 1) + datetime.timedelta(days=i % 365),
                    'c_json': None if i % 7 == 0 else {'i': i, 'tag': f't{i}'},
                    'c_bytes': None if i % 7 == 0 else f'b{i}'.encode(),
                }
                for i in range(n)
            ],
        )

        tbl = import_sql(src, engine, 'imported')

        # Schema: types + nullability propagated from SA columns
        schema = tbl._get_schema()
        assert set(schema) == {'c_int', 'c_str', 'c_float', 'c_bool', 'c_ts', 'c_date', 'c_json', 'c_bytes'}
        assert schema['c_int'].is_int_type() and not schema['c_int'].nullable
        assert schema['c_str'].is_string_type() and not schema['c_str'].nullable
        assert schema['c_float'].is_float_type() and schema['c_float'].nullable
        assert schema['c_bool'].is_bool_type() and schema['c_bool'].nullable
        assert schema['c_ts'].is_timestamp_type() and schema['c_ts'].nullable
        assert schema['c_date'].is_date_type() and schema['c_date'].nullable
        assert schema['c_json'].is_json_type() and schema['c_json'].nullable
        assert schema['c_bytes'].is_binary_type() and schema['c_bytes'].nullable

        # Single-version-bump claim: streaming N=2500 rows still results in version 1
        assert tbl._tbl_version.get().version == 1

        # Row count + values + NULL roundtrip
        result = tbl.order_by(tbl.c_int).select(
            tbl.c_int, tbl.c_str, tbl.c_float, tbl.c_bool, tbl.c_ts, tbl.c_date, tbl.c_json, tbl.c_bytes
        ).collect()
        assert len(result) == n
        for i, row in enumerate(result):
            assert row['c_int'] == i
            assert row['c_str'] == f'row_{i}'
            if i % 7 == 0:
                assert row['c_float'] is None
                assert row['c_bool'] is None
                assert row['c_ts'] is None
                assert row['c_date'] is None
                assert row['c_json'] is None
                assert row['c_bytes'] is None
            else:
                assert row['c_float'] == float(i) * 1.5
                assert row['c_bool'] == (i % 2 == 0)
                assert row['c_date'] == datetime.date(2024, 1, 1) + datetime.timedelta(days=i % 365)
                assert row['c_json'] == {'i': i, 'tag': f't{i}'}
                assert row['c_bytes'] == f'b{i}'.encode()
