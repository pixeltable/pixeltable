import dataclasses
import datetime
import json
import pathlib
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable

import PIL.Image
import pytest
import sqlalchemy as sql
import sqlalchemy.dialects.postgresql
import sqlalchemy.dialects.sqlite  # noqa: F401

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.io.sql import export_sql, import_sql

from ..utils import error, get_documents, get_image_files, get_video_files, pxt_raises


@dataclasses.dataclass(frozen=True)
class _DialectSpec:
    name: str
    connect: Callable[[pathlib.Path], str]  # tmp_path -> connection string
    sa_types: dict[str, type]  # pxt col -> SA type for create-table assertions
    decode: dict[str, Callable[[Any], Any]]  # SA value -> python value comparable to rows[i][col]
    # if_exists='insert' requires reflecting the target schema, which roundtrips through SA types;
    # sqlite's UUID -> NUMERIC mismatch makes the compat probe fail, so skip it there.
    supports_insert: bool = True


_IMPORT_DBMS = ['sqlite', 'postgresql']


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
    if len(rows) > 0:
        with engine.begin() as conn:
            conn.execute(src.insert(), rows)
    return src


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


    @pytest.mark.parametrize('dialect', _IMPORT_DBMS)
    def test_import_full_table(self, uses_db: None, tmp_path: pathlib.Path, dialect: str) -> None:
        """End-to-end import of a full SA Table: type inference for all common SA types, nullable vs non-nullable
        propagation, NULL-to-None value roundtrip, exact value preservation, and the single-version-bump claim
        across batch boundaries (rows > BATCH_SIZE so streaming actually engages)."""
        engine = _import_engine(dialect, tmp_path)
        n = 2500  # > SqlDataNode.BATCH_SIZE (1024) to force at least 3 batches

        seed_rows = [
            {
                'c_int': i,
                'c_str': f'row_{i}',
                # every 7th row uses NULL so we exercise both NULL and value paths
                'c_float': None if i % 7 == 0 else float(i) * 1.5,
                'c_bool': None if i % 7 == 0 else (i % 2 == 0),
                'c_ts': None if i % 7 == 0 else datetime.datetime(2024, 1, 1) + datetime.timedelta(seconds=i),
                'c_date': None if i % 7 == 0 else datetime.date(2024, 1, 1) + datetime.timedelta(days=i % 365),
                'c_uuid': None if i % 7 == 0 else uuid.uuid5(uuid.NAMESPACE_DNS, f'row_{i}'),
                'c_json': None if i % 7 == 0 else {'i': i, 'tag': f't{i}'},
                'c_bytes': None if i % 7 == 0 else f'b{i}'.encode(),
            }
            for i in range(n)
        ]

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
                sql.Column('c_uuid', sql.Uuid, nullable=True),
                sql.Column('c_json', sql.JSON, nullable=True),
                sql.Column('c_bytes', sql.LargeBinary, nullable=True),
            ],
            seed_rows,
        )

        tbl = import_sql(src, engine, 'imported')

        # Schema: types + nullability propagated from SA columns
        meta = tbl.get_metadata()
        cols = meta['columns']
        assert set(cols) == {'c_int', 'c_str', 'c_float', 'c_bool', 'c_ts', 'c_date', 'c_uuid', 'c_json', 'c_bytes'}
        assert cols['c_int']['type_'] == 'Required[Int]'
        assert cols['c_str']['type_'] == 'Required[String]'
        assert cols['c_float']['type_'] == 'Float'
        assert cols['c_bool']['type_'] == 'Bool'
        assert cols['c_ts']['type_'] == 'Timestamp'
        assert cols['c_date']['type_'] == 'Date'
        assert cols['c_uuid']['type_'] == 'UUID'
        assert cols['c_json']['type_'] == 'Json'
        assert cols['c_bytes']['type_'] == 'Binary'

        # Single-version-bump claim: streaming N=2500 rows still results in version 1
        assert meta['version'] == 1

        # Row count + values + NULL roundtrip
        result = (
            tbl.order_by(tbl.c_int)
            .select(
                tbl.c_int,
                tbl.c_str,
                tbl.c_float,
                tbl.c_bool,
                tbl.c_date,
                tbl.c_uuid,
                tbl.c_json,
                tbl.c_bytes,
                c_ts=tbl.c_ts.strip_timezone(),
            )
            .collect()
        )
        assert len(result) == len(seed_rows)
        assert list(result) == seed_rows

    @pytest.mark.parametrize('dialect', _IMPORT_DBMS)
    def test_import_select_and_filter(self, uses_db: None, tmp_path: pathlib.Path, dialect: str) -> None:
        """Import via `sa.select(...)` rather than a bare Table: column projection (subset), row filter via
        `.where(...)`, labeled expressions, accepting an `sa.Connection` (not just an Engine), and the 0-row
        edge case (impossible filter -> empty destination, schema still created)."""
        engine = _import_engine(dialect, tmp_path)
        rows = [{'c_int': i, 'c_str': f'row_{i}', 'c_float': float(i)} for i in range(20)]
        src = _seed_source(
            engine,
            'src_proj',
            [
                sql.Column('c_int', sql.Integer, nullable=False),
                sql.Column('c_str', sql.String, nullable=False),
                sql.Column('c_float', sql.Float, nullable=False),
            ],
            rows,
        )

        # Projection + filter + labeled SA function with explicit type annotation. Pass a Connection
        # (caller-managed lifecycle).
        with engine.connect() as conn:
            stmt = sql.select(src.c.c_int, sql.func.upper(src.c.c_str, type_=sql.String).label('c_upper')).where(
                src.c.c_int >= 10
            )
            tbl = import_sql(stmt, conn, 'projected')

        # Only projected columns landed in the destination; c_float must NOT be present.
        cols = tbl.get_metadata()['columns']
        assert set(cols) == {'c_int', 'c_upper'}
        assert 'String' in cols['c_upper']['type_']
        result = tbl.order_by(tbl.c_int).select(tbl.c_int, tbl.c_upper).collect()
        assert len(result) == 10
        for j, row in enumerate(result):
            i = j + 10
            assert row['c_int'] == i
            assert row['c_upper'] == f'ROW_{i}'

        # 0-row import: same source, impossible filter. Destination table is still created with the right schema.
        empty_stmt = sql.select(src.c.c_int, src.c.c_str).where(src.c.c_int < 0)
        empty_tbl = import_sql(empty_stmt, engine, 'empty_proj')
        empty_cols = empty_tbl.get_metadata()['columns']
        assert set(empty_cols) == {'c_int', 'c_str'}
        assert empty_tbl.count() == 0

    @pytest.mark.parametrize('dialect', _IMPORT_DBMS)
    def test_media_via_overrides(self, uses_db: None, tmp_path: pathlib.Path, dialect: str) -> None:
        """`schema_overrides` promotes plain String path columns into Pixeltable media types (Image, Video,
        Document)"""
        engine = _import_engine(dialect, tmp_path)
        img_paths = get_image_files()[:2]
        video_paths = get_video_files(include_vfr=False, include_mpgs=False, extension='.mp4')[:2]
        doc_paths = [p for p in get_documents() if p.endswith('.pdf')][:2]
        assert len(img_paths) == 2 and len(video_paths) == 2 and len(doc_paths) == 2

        rows = [
            {'c_int': i, 'c_img': img_paths[i], 'c_vid': video_paths[i], 'c_doc': doc_paths[i], 'c_path': img_paths[i]}
            for i in range(2)
        ]
        src = _seed_source(
            engine,
            'src_media',
            [
                sql.Column('c_int', sql.Integer, nullable=False),
                sql.Column('c_img', sql.String, nullable=False),
                sql.Column('c_vid', sql.String, nullable=False),
                sql.Column('c_doc', sql.String, nullable=False),
                sql.Column('c_path', sql.String, nullable=False),
            ],
            rows,
        )

        tbl = import_sql(
            src, engine, 'media_dest', schema_overrides={'c_img': pxt.Image, 'c_vid': pxt.Video, 'c_doc': pxt.Document}
        )

        # Schema reflects the overrides; non-overridden c_path remains String.
        cols = tbl.get_metadata()['columns']
        assert 'Image' in cols['c_img']['type_']
        assert 'Video' in cols['c_vid']['type_']
        assert 'Document' in cols['c_doc']['type_']
        assert 'String' in cols['c_path']['type_']

        # Collected Image cells are PIL Images (the media is decodable).
        img_result = tbl.order_by(tbl.c_int).select(tbl.c_img).collect()
        assert len(img_result) == 2
        for row in img_result:
            assert isinstance(row['c_img'], PIL.Image.Image)

        # Media columns expose `.fileurl`. Pixeltable references local paths in place (only external URLs are
        # pulled into the file cache via CachePrefetchNode; local paths skip prefetch entirely). So each fileurl
        # must resolve to the exact source path we inserted, proving the path round-tripped correctly through
        # SqlDataNode -> InsertableTable.insert -> store.
        urls = (
            tbl.order_by(tbl.c_int)
            .select(c_img=tbl.c_img.fileurl, c_vid=tbl.c_vid.fileurl, c_doc=tbl.c_doc.fileurl)
            .collect()
        )
        for i, row in enumerate(urls):
            for col_name, expected in (('c_img', img_paths[i]), ('c_vid', video_paths[i]), ('c_doc', doc_paths[i])):
                url = row[col_name]
                assert isinstance(url, str) and url.startswith('file://'), (col_name, url)
                # url2pathname handles both percent-decoding and the OS-specific URL-to-path conversion
                # (eg, on Windows '/D:/a/foo' -> 'D:\\a\\foo'); pathlib.Path comparison is OS-agnostic.
                actual = pathlib.Path(urllib.request.url2pathname(urllib.parse.urlparse(url).path))
                assert actual == pathlib.Path(expected), (col_name, actual, expected)

        # Control column: kept as String, value is the raw path we inserted.
        path_result = tbl.order_by(tbl.c_int).select(tbl.c_path).collect()
        for i, row in enumerate(path_result):
            assert row['c_path'] == img_paths[i]

    def test_if_exists(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Walk the if_exists matrix in a single test, since the branching is purely pixeltable-side and doesn't
        depend on the SQL backend."""
        engine = _import_engine('sqlite', tmp_path)
        seed_a = [{'c_int': i, 'c_str': f'a_{i}'} for i in range(3)]
        seed_b = [{'c_int': 100 + i, 'c_str': f'b_{i}'} for i in range(2)]
        src_a = _seed_source(
            engine,
            'src_a',
            [sql.Column('c_int', sql.Integer, nullable=False), sql.Column('c_str', sql.String, nullable=False)],
            seed_a,
        )
        src_b = _seed_source(
            engine,
            'src_b',
            [sql.Column('c_int', sql.Integer, nullable=False), sql.Column('c_str', sql.String, nullable=False)],
            seed_b,
        )

        # if_exists='error', table doesn't exist -> succeeds.
        tbl = import_sql(src_a, engine, 'dest')
        assert tbl.count() == len(seed_a)

        # if_exists='error', table exists -> rejects.
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='existing table'):
            import_sql(src_b, engine, 'dest')
        # destination must be untouched after the rejection
        assert tbl.count() == len(seed_a)

        # if_exists='append' against missing table -> rejects (pxt.get_table raises PATH_NOT_FOUND).
        with pxt_raises(pxt.ErrorCode.PATH_NOT_FOUND, match='never_existed'):
            import_sql(src_a, engine, 'never_existed', if_exists='append')

        # if_exists='append' happy path -> preserves existing rows and adds new ones.
        import_sql(src_b, engine, 'dest', if_exists='append')
        result = tbl.order_by(tbl.c_int).select(tbl.c_int, tbl.c_str).collect()
        assert [(r['c_int'], r['c_str']) for r in result] == [
            *((r['c_int'], r['c_str']) for r in seed_a),
            *((r['c_int'], r['c_str']) for r in seed_b),
        ]

        # if_exists='append' with a source column not present in destination -> COLUMN_NOT_FOUND.
        src_extra = _seed_source(
            engine,
            'src_extra',
            [
                sql.Column('c_int', sql.Integer, nullable=False),
                sql.Column('c_str', sql.String, nullable=False),
                sql.Column('c_unknown', sql.Integer, nullable=False),
            ],
            [{'c_int': 9, 'c_str': 'x', 'c_unknown': 1}],
        )
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='c_unknown'):
            import_sql(src_extra, engine, 'dest', if_exists='append')

        # if_exists='append' with incompatible source column type vs destination -> TYPE_MISMATCH.
        # Build a source whose c_int is a String, attempt to append into the existing Int column.
        src_mismatch = _seed_source(
            engine,
            'src_mismatch',
            [sql.Column('c_int', sql.String, nullable=False), sql.Column('c_str', sql.String, nullable=False)],
            [{'c_int': 'not_an_int', 'c_str': 'y'}],
        )
        with pxt_raises(pxt.ErrorCode.TYPE_MISMATCH, match='c_int'):
            import_sql(src_mismatch, engine, 'dest', if_exists='append')

        # if_exists='replace' is the documented foot-gun and is rejected with a migration hint pointing the user
        # at drop_table + if_exists='error'.
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'drop_table'):
            import_sql(src_a, engine, 'dest', if_exists='replace')  # type: ignore[arg-type]

        # any other string is also rejected.
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r"must be one of 'error', 'append'"):
            import_sql(src_a, engine, 'dest', if_exists='garbage')  # type: ignore[arg-type]

    def test_validation_errors(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Broad sweep of the remaining `RequestError` / `NotFoundError` paths in `import_sql` and
        `SqlDataNode._open()`.
        """
        engine = _import_engine('sqlite', tmp_path)

        # `sql.Interval` has no entry in our SA -> pxt mapping; without an override the inference must fail.
        meta = sql.MetaData()
        bad_type_src = sql.Table(
            'bad_type',
            meta,
            sql.Column('c_int', sql.Integer, nullable=False),
            sql.Column('c_dur', sql.Interval, nullable=False),
        )
        meta.create_all(engine)
        with pxt_raises(pxt.ErrorCode.INVALID_TYPE, match='c_dur'):
            import_sql(bad_type_src, engine, 'bad_type_dest')

        # The same source becomes valid once schema_overrides resolves the unmappable column.
        tbl_overridden = import_sql(bad_type_src, engine, 'bad_type_dest_ok', schema_overrides={'c_dur': pxt.String})
        assert 'String' in tbl_overridden.get_metadata()['columns']['c_dur']['type_']

        ok_src = _seed_source(
            engine,
            'ok_src',
            [sql.Column('c_int', sql.Integer, nullable=False), sql.Column('c_str', sql.String, nullable=False)],
            [{'c_int': 1, 'c_str': 'a'}],
        )
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='nonexistent'):
            import_sql(ok_src, engine, 'never1', schema_overrides={'nonexistent': pxt.Int})

        # Two expressions aliased to the same name -> import_sql rejects up-front.
        dup_stmt = sql.select(ok_src.c.c_int.label('c_int'), (ok_src.c.c_int + 1).label('c_int'))
        with pxt_raises(pxt.ErrorCode.INVALID_SCHEMA, match='duplicate output column'):
            import_sql(dup_stmt, engine, 'dup_dest')

        # Pre-create a destination with a computed column, then append a source that supplies a value for it.
        comp_dest = pxt.create_table('comp_dest', {'c_int': pxt.Int})
        comp_dest.add_computed_column(c_doubled=comp_dest.c_int * 2)
        comp_src = _seed_source(
            engine,
            'comp_src',
            [sql.Column('c_int', sql.Integer, nullable=False), sql.Column('c_doubled', sql.Integer, nullable=False)],
            [{'c_int': 5, 'c_doubled': 10}],
        )
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='computed column'):
            import_sql(comp_src, engine, 'comp_dest', if_exists='append')

        # Pre-create a destination where c_required is non-nullable; append a source that doesn't supply it.
        pxt.create_table('req_dest', {'c_int': pxt.Int, 'c_required': pxt.Required[pxt.String]})
        partial_src = _seed_source(
            engine, 'partial_src', [sql.Column('c_int', sql.Integer, nullable=False)], [{'c_int': 1}]
        )
        with pxt_raises(pxt.ErrorCode.MISSING_REQUIRED, match='c_required'):
            import_sql(partial_src, engine, 'req_dest', if_exists='append')

    def test_import_on_error_ignore(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """`on_error='ignore'` import: per-row computed-column failures surface as nulls."""
        engine = _import_engine('sqlite', tmp_path)

        dest = pxt.create_table('on_error_dest', {'c_int': pxt.Int})
        dest.add_computed_column(c_checked=error(dest.c_int < 0))

        values = [1, -1, 2, -2, 3]
        src = _seed_source(
            engine,
            'on_error_src',
            [sql.Column('c_int', sql.Integer, nullable=False)],
            [{'c_int': v} for v in values],
        )

        import_sql(src, engine, 'on_error_dest', if_exists='append', on_error='ignore')
        result = dest.order_by(dest.c_int).select(dest.c_int, dest.c_checked).collect()
        assert result['c_int'] == sorted(values)
        assert result['c_checked'] == [None if v < 0 else False for v in sorted(values)]
