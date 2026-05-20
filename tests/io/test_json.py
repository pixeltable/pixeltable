import datetime
import json
import pathlib

import pytest

import pixeltable as pxt
from pixeltable.config import Config

from ..utils import create_all_datatypes_tbl, get_image_files, skip_test_if_not_installed, validate_update_status


class TestJson:
    def test_export_all_types(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Export a table with every supported type and verify the JSONL output."""
        t = create_all_datatypes_tbl()
        rows = t.collect()

        json_path = tmp_path / 'all_types.jsonl'
        pxt.io.export_json(t, json_path)

        with open(json_path, encoding='utf-8') as f:
            exported = [json.loads(line) for line in f]

        assert len(exported) == len(rows)

        for exp_row, orig_row in zip(exported, rows):
            assert exp_row['c_string'] == orig_row['c_string']
            assert exp_row['c_int'] == orig_row['c_int']
            assert exp_row['c_float'] == orig_row['c_float']
            assert exp_row['c_bool'] == orig_row['c_bool']

            assert isinstance(exp_row['c_timestamp'], str)
            assert isinstance(exp_row['c_date'], str)
            assert datetime.date.fromisoformat(exp_row['c_date']) == orig_row['c_date']

            assert exp_row['c_uuid'] == str(orig_row['c_uuid'])
            assert exp_row['c_json'] == orig_row['c_json']
            assert exp_row['c_array'] == orig_row['c_array'].tolist()

            for col in ['c_image', 'c_video', 'c_audio', 'c_document']:
                assert isinstance(exp_row[col], str), f'{col} should be a string'
                assert exp_row[col] != '', f'{col} should not be empty'

        # Verify media columns export the authoritative file URL, not a cached path
        media_cols = {'c_image': t.c_image, 'c_video': t.c_video, 'c_audio': t.c_audio, 'c_document': t.c_document}
        fileurls = t.select(*[col.fileurl for col in media_cols.values()]).collect()
        for exp_row, url_row in zip(exported, fileurls):
            for col_name in media_cols:
                assert exp_row[col_name] == url_row[f'{col_name}_fileurl']

    def test_export_non_serializable_json_errors(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Exporting a JSON column with non-serializable values should raise an error."""
        t = create_all_datatypes_tbl(non_serializable_json=True)
        with pytest.raises(pxt.Error, match='not JSON-serializable'):
            pxt.io.export_json(t, tmp_path / 'should_fail.jsonl')

    def test_export_with_nulls(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Verify null handling across multiple types."""
        t = pxt.create_table(
            'test_json_nulls',
            {
                'c_int': pxt.Int,
                'c_string': pxt.String,
                'c_float': pxt.Float,
                'c_json': pxt.Json,
                'c_timestamp': pxt.Timestamp,
            },
        )
        t.insert(
            [
                {'c_int': 1, 'c_string': None, 'c_float': None, 'c_json': None, 'c_timestamp': None},
                {'c_int': None, 'c_string': 'hello', 'c_float': 1.5, 'c_json': {'a': 1}, 'c_timestamp': None},
            ]
        )

        json_path = tmp_path / 'nulls.jsonl'
        pxt.io.export_json(t, json_path)

        with open(json_path, encoding='utf-8') as f:
            exported = [json.loads(line) for line in f]

        assert exported[0]['c_string'] is None
        assert exported[0]['c_float'] is None
        assert exported[0]['c_json'] is None
        assert exported[0]['c_timestamp'] is None
        assert exported[1]['c_int'] is None

    def test_export_with_query(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test export with filtering and column selection."""
        t = pxt.create_table('test_json_query', {'c_int': pxt.Int, 'c_string': pxt.String})
        rows = [{'c_int': i, 'c_string': f'row_{i}'} for i in range(10)]
        validate_update_status(t.insert(rows), expected_rows=10)

        # Filtered
        json_path = tmp_path / 'filtered.jsonl'
        pxt.io.export_json(t.where(t.c_int < 5), json_path)
        with open(json_path, encoding='utf-8') as f:
            assert len([json.loads(line) for line in f]) == 5

        # Column subset
        json_path2 = tmp_path / 'subset.jsonl'
        pxt.io.export_json(t.select(t.c_string), json_path2)
        with open(json_path2, encoding='utf-8') as f:
            exported = [json.loads(line) for line in f]
        assert len(exported) == 10
        assert list(exported[0].keys()) == ['c_string']

    def test_export_non_ascii(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Verify non-ASCII characters are preserved."""
        t = pxt.create_table('test_json_encoding', {'name': pxt.String})
        t.insert([{'name': 'Manwë'}, {'name': 'Fëanor'}])

        json_path = tmp_path / 'encoding.jsonl'
        pxt.io.export_json(t, json_path)

        with open(json_path, encoding='utf-8') as f:
            exported = [json.loads(line) for line in f]
        names = [row['name'] for row in exported]
        assert 'Manwë' in names
        assert 'Fëanor' in names

    def test_round_trip(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Export JSONL, re-import, and verify data matches."""
        t = pxt.create_table('test_json_rt', {'c_int': pxt.Int, 'c_string': pxt.String, 'c_float': pxt.Float})
        t.insert([{'c_int': 1, 'c_string': 'hello', 'c_float': 1.5}, {'c_int': 2, 'c_string': 'world', 'c_float': 2.5}])

        json_path = tmp_path / 'round_trip.jsonl'
        pxt.io.export_json(t, json_path)

        t2 = pxt.io.import_json('test_json_rt2', str(json_path))

        original = t.order_by(t.c_int).collect()
        reimported = t2.order_by(t2.c_int).collect()

        assert original == reimported

    def test_round_trip_media(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Export JSONL with media columns, re-import, and verify file URLs survive the round-trip."""
        t = create_all_datatypes_tbl()

        json_path = tmp_path / 'round_trip_media.jsonl'
        pxt.io.export_json(t, json_path)

        # Build schema overrides from the original table, excluding binary (not exportable to JSON)
        schema_overrides = {name: ct for name, ct in t._get_schema().items() if not ct.is_binary_type()}
        t2 = pxt.io.import_json('test_json_rt_media', str(json_path), schema_overrides=schema_overrides)

        # Select only columns that survive JSON export (binary is excluded)
        exportable_cols = [getattr(t, name) for name in t2.columns()]
        original = t.select(*exportable_cols).order_by(t.row_id).collect()
        reimported = t2.order_by(t2.row_id).collect()

        assert original == reimported

    def test_export_remote_urls(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Verify that remote URLs (S3, HTTPS) are exported as-is."""
        skip_test_if_not_installed('boto3')
        urls = {
            'c_video': 's3://multimedia-commons/data/videos/mp4/ffe/ff3/ffeff3c6bf57504e7a6cecaff6aefbc9.mp4',
            'c_audio': 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/tests/data/audio/sample.flac',
            'c_document': 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/tests/data/documents/1706.03762.pdf',
        }

        t = pxt.create_table(
            'test_json_remote', {'c_video': pxt.Video, 'c_audio': pxt.Audio, 'c_document': pxt.Document}
        )
        t.insert([urls])

        json_path = tmp_path / 'remote.jsonl'
        pxt.io.export_json(t, json_path)

        with open(json_path, encoding='utf-8') as f:
            exported = [json.loads(line) for line in f]

        for col, expected_url in urls.items():
            assert exported[0][col] == expected_url, f'{col}: expected {expected_url}, got {exported[0][col]}'

    def test_export_unstored_media_expression_errors(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Exporting a media-typed expression that is not backed by a stored column should raise an error."""

        t = pxt.create_table('test_json_transform', {'img': pxt.Image})
        t.insert([{'img': get_image_files()[0]}])

        with pytest.raises(pxt.Error, match='Cannot export media expression'):
            pxt.io.export_json(t.select(t.img.rotate(90)), tmp_path / 'should_fail.jsonl')

    def test_export_computed_media_with_destination(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Computed media columns with a local file destination export their file URLs."""
        dest_path = Config.get().home / 'test-json-dest'
        dest_path.mkdir(parents=True, exist_ok=True)
        dest_uri = dest_path.as_uri()

        t = pxt.create_table('test_json_computed_dest', {'img': pxt.Image})
        t.add_computed_column(rotated=t.img.rotate(90), destination=dest_uri)
        t.insert([{'img': f} for f in get_image_files()[:3]])

        json_path = tmp_path / 'computed_dest.jsonl'
        pxt.io.export_json(t, json_path)

        with open(json_path, encoding='utf-8') as f:
            exported = [json.loads(line) for line in f]

        assert len(exported) == 3
        fileurls = t.select(t.rotated.fileurl).collect()
        for exp_row, url_row in zip(exported, fileurls):
            assert exp_row['rotated'] == url_row['rotated_fileurl']
            assert exp_row['rotated'].startswith('file:')
            assert dest_path.as_posix() in exp_row['rotated']
