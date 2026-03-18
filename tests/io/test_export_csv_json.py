import csv
import datetime
import json
import pathlib
import uuid

import numpy as np

import pixeltable as pxt

from ..utils import get_image_files, validate_update_status


class TestExportCsv:
    def test_export_csv_basic(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test basic CSV export with common column types."""
        t = pxt.create_table(
            'test_csv',
            {'c_int': pxt.Int, 'c_string': pxt.String, 'c_float': pxt.Float, 'c_bool': pxt.Bool, 'c_json': pxt.Json},
        )
        json_val1 = {'key': 'value1', 'num': 1}
        json_val2 = {'key': 'value2', 'num': 2}
        rows = [
            {'c_int': 1, 'c_string': 'hello', 'c_float': 1.5, 'c_bool': True, 'c_json': json_val1},
            {'c_int': 2, 'c_string': 'world', 'c_float': 2.5, 'c_bool': False, 'c_json': json_val2},
        ]
        validate_update_status(t.insert(rows), expected_rows=2)

        csv_path = tmp_path / 'test.csv'
        pxt.io.export_csv(t, csv_path)
        assert csv_path.exists()

        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            exported_rows = list(reader)

        assert len(exported_rows) == 2
        assert set(exported_rows[0].keys()) == {'c_int', 'c_string', 'c_float', 'c_bool', 'c_json'}

        # CSV values are strings — verify round-trip
        for exported, original in zip(exported_rows, rows):
            assert int(exported['c_int']) == original['c_int']
            assert exported['c_string'] == original['c_string']
            assert float(exported['c_float']) == original['c_float']
            assert json.loads(exported['c_json']) == original['c_json']

    def test_export_csv_with_query(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test CSV export with a filtered query and column subset."""
        t = pxt.create_table('test_csv_query', {'c_int': pxt.Int, 'c_string': pxt.String})
        rows = [{'c_int': i, 'c_string': f'row_{i}'} for i in range(10)]
        validate_update_status(t.insert(rows), expected_rows=10)

        # Export filtered query
        csv_path = tmp_path / 'filtered.csv'
        pxt.io.export_csv(t.where(t.c_int < 5), csv_path)

        with open(csv_path, encoding='utf-8') as f:
            exported_rows = list(csv.DictReader(f))
        assert len(exported_rows) == 5

        # Export column subset
        csv_path2 = tmp_path / 'subset.csv'
        pxt.io.export_csv(t.select(t.c_string), csv_path2)

        with open(csv_path2, encoding='utf-8') as f:
            exported_rows2 = list(csv.DictReader(f))
        assert len(exported_rows2) == 10
        assert list(exported_rows2[0].keys()) == ['c_string']

    def test_export_csv_timestamps(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test CSV export with timestamp and date columns."""
        from zoneinfo import ZoneInfo

        tz = ZoneInfo('America/New_York')
        ts1 = datetime.datetime(2024, 1, 15, 10, 30, 0, tzinfo=tz)
        d1 = datetime.date(2024, 6, 15)

        t = pxt.create_table('test_csv_ts', {'c_ts': pxt.Timestamp, 'c_date': pxt.Date})
        t.insert([{'c_ts': ts1, 'c_date': d1}])

        csv_path = tmp_path / 'timestamps.csv'
        pxt.io.export_csv(t, csv_path)

        with open(csv_path, encoding='utf-8') as f:
            exported_rows = list(csv.DictReader(f))
        assert len(exported_rows) == 1
        # Values should be non-empty strings
        assert exported_rows[0]['c_ts'] != ''
        assert exported_rows[0]['c_date'] != ''

    def test_export_csv_array(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test CSV export with array columns (serialized as JSON)."""
        t = pxt.create_table('test_csv_array', {'c_int': pxt.Int, 'c_arr': pxt.Array[(3,), np.float64]})  # type: ignore[misc]
        rows = [{'c_int': i, 'c_arr': np.array([i * 1.0, i * 2.0, i * 3.0])} for i in range(5)]
        validate_update_status(t.insert(rows), expected_rows=5)

        csv_path = tmp_path / 'arrays.csv'
        pxt.io.export_csv(t, csv_path)

        with open(csv_path, encoding='utf-8') as f:
            exported_rows = list(csv.DictReader(f))
        assert len(exported_rows) == 5
        for i, row in enumerate(exported_rows):
            arr = json.loads(row['c_arr'])
            assert arr == [i * 1.0, i * 2.0, i * 3.0]

    def test_export_csv_media(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test that media columns are exported as file path strings."""
        image_files = get_image_files()
        t = pxt.create_table('test_csv_media', {'c_int': pxt.Int, 'c_img': pxt.Image})
        t.insert([{'c_int': 1, 'c_img': image_files[0]}, {'c_int': 2, 'c_img': image_files[1]}])

        csv_path = tmp_path / 'media.csv'
        pxt.io.export_csv(t, csv_path)

        with open(csv_path, encoding='utf-8') as f:
            exported_rows = list(csv.DictReader(f))
        assert len(exported_rows) == 2
        assert 'c_img' in exported_rows[0]
        # Media columns should be non-empty path strings
        for row in exported_rows:
            assert row['c_img'] != ''
            assert isinstance(row['c_img'], str)

    def test_export_csv_custom_delimiter(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test CSV export with custom delimiter."""
        t = pxt.create_table('test_csv_delim', {'c_int': pxt.Int, 'c_string': pxt.String})
        t.insert([{'c_int': 1, 'c_string': 'hello'}])

        csv_path = tmp_path / 'tab.csv'
        pxt.io.export_csv(t, csv_path, delimiter='\t')

        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            exported_rows = list(reader)
        assert len(exported_rows) == 1
        assert int(exported_rows[0]['c_int']) == 1
        assert exported_rows[0]['c_string'] == 'hello'

    def test_export_csv_nulls(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test CSV export with null values."""
        t = pxt.create_table('test_csv_nulls', {'c_int': pxt.Int, 'c_string': pxt.String})
        t.insert([{'c_int': 1, 'c_string': None}, {'c_int': None, 'c_string': 'hello'}])

        csv_path = tmp_path / 'nulls.csv'
        pxt.io.export_csv(t, csv_path)

        with open(csv_path, encoding='utf-8') as f:
            exported_rows = list(csv.DictReader(f))
        assert len(exported_rows) == 2
        # Null values should be empty strings in CSV
        assert exported_rows[0]['c_string'] == ''
        assert exported_rows[1]['c_int'] == ''


class TestExportJson:
    def test_export_json_basic(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test basic JSON export with common column types."""
        t = pxt.create_table(
            'test_json',
            {'c_int': pxt.Int, 'c_string': pxt.String, 'c_float': pxt.Float, 'c_bool': pxt.Bool, 'c_json': pxt.Json},
        )
        json_val1 = {'key': 'value1', 'num': 1}
        json_val2 = {'key': 'value2', 'num': 2}
        rows = [
            {'c_int': 1, 'c_string': 'hello', 'c_float': 1.5, 'c_bool': True, 'c_json': json_val1},
            {'c_int': 2, 'c_string': 'world', 'c_float': 2.5, 'c_bool': False, 'c_json': json_val2},
        ]
        validate_update_status(t.insert(rows), expected_rows=2)

        json_path = tmp_path / 'test.json'
        pxt.io.export_json(t, json_path)
        assert json_path.exists()

        with open(json_path, encoding='utf-8') as f:
            exported = json.load(f)

        assert isinstance(exported, list)
        assert len(exported) == 2
        for exp_row, orig_row in zip(exported, rows):
            assert exp_row['c_int'] == orig_row['c_int']
            assert exp_row['c_string'] == orig_row['c_string']
            assert exp_row['c_float'] == orig_row['c_float']
            assert exp_row['c_bool'] == orig_row['c_bool']
            assert exp_row['c_json'] == orig_row['c_json']

    def test_export_json_with_query(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test JSON export with a filtered query and column subset."""
        t = pxt.create_table('test_json_query', {'c_int': pxt.Int, 'c_string': pxt.String})
        rows = [{'c_int': i, 'c_string': f'row_{i}'} for i in range(10)]
        validate_update_status(t.insert(rows), expected_rows=10)

        # Export filtered query
        json_path = tmp_path / 'filtered.json'
        pxt.io.export_json(t.where(t.c_int < 5), json_path)

        with open(json_path, encoding='utf-8') as f:
            exported = json.load(f)
        assert len(exported) == 5

        # Export column subset
        json_path2 = tmp_path / 'subset.json'
        pxt.io.export_json(t.select(t.c_string), json_path2)

        with open(json_path2, encoding='utf-8') as f:
            exported2 = json.load(f)
        assert len(exported2) == 10
        assert list(exported2[0].keys()) == ['c_string']

    def test_export_json_timestamps(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test JSON export with timestamp and date columns produces ISO strings."""
        from zoneinfo import ZoneInfo

        tz = ZoneInfo('America/New_York')
        ts1 = datetime.datetime(2024, 1, 15, 10, 30, 0, tzinfo=tz)
        d1 = datetime.date(2024, 6, 15)

        t = pxt.create_table('test_json_ts', {'c_ts': pxt.Timestamp, 'c_date': pxt.Date})
        t.insert([{'c_ts': ts1, 'c_date': d1}])

        json_path = tmp_path / 'timestamps.json'
        pxt.io.export_json(t, json_path)

        with open(json_path, encoding='utf-8') as f:
            exported = json.load(f)
        assert len(exported) == 1
        # Should be valid ISO format strings
        assert isinstance(exported[0]['c_ts'], str)
        assert isinstance(exported[0]['c_date'], str)
        assert '2024' in exported[0]['c_ts']
        assert exported[0]['c_date'] == '2024-06-15'

    def test_export_json_uuid(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test JSON export serializes UUIDs as strings."""
        t = pxt.create_table('test_json_uuid', {'c_uuid': pxt.UUID})
        test_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, 'test')
        t.insert([{'c_uuid': test_uuid}])

        json_path = tmp_path / 'uuid.json'
        pxt.io.export_json(t, json_path)

        with open(json_path, encoding='utf-8') as f:
            exported = json.load(f)
        assert exported[0]['c_uuid'] == str(test_uuid)

    def test_export_json_array(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test JSON export with array columns."""
        t = pxt.create_table('test_json_array', {'c_int': pxt.Int, 'c_arr': pxt.Array[(3,), np.float64]})  # type: ignore[misc]
        rows = [{'c_int': i, 'c_arr': np.array([i * 1.0, i * 2.0, i * 3.0])} for i in range(5)]
        validate_update_status(t.insert(rows), expected_rows=5)

        json_path = tmp_path / 'arrays.json'
        pxt.io.export_json(t, json_path)

        with open(json_path, encoding='utf-8') as f:
            exported = json.load(f)
        assert len(exported) == 5
        for i, row in enumerate(exported):
            assert row['c_arr'] == [i * 1.0, i * 2.0, i * 3.0]

    def test_export_json_media(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test that media columns are exported as file path strings."""
        image_files = get_image_files()
        t = pxt.create_table('test_json_media', {'c_int': pxt.Int, 'c_img': pxt.Image})
        t.insert([{'c_int': 1, 'c_img': image_files[0]}, {'c_int': 2, 'c_img': image_files[1]}])

        json_path = tmp_path / 'media.json'
        pxt.io.export_json(t, json_path)

        with open(json_path, encoding='utf-8') as f:
            exported = json.load(f)
        assert len(exported) == 2
        for row in exported:
            assert isinstance(row['c_img'], str)
            assert row['c_img'] != ''

    def test_export_json_indent(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test JSON export with pretty-printing."""
        t = pxt.create_table('test_json_indent', {'c_int': pxt.Int})
        t.insert([{'c_int': 1}])

        # Compact
        compact_path = tmp_path / 'compact.json'
        pxt.io.export_json(t, compact_path)
        compact_content = compact_path.read_text()
        assert '\n' not in compact_content.strip() or compact_content.count('\n') <= 1

        # Pretty-printed
        pretty_path = tmp_path / 'pretty.json'
        pxt.io.export_json(t, pretty_path, indent=2)
        pretty_content = pretty_path.read_text()
        assert pretty_content.count('\n') > 1

    def test_export_json_nulls(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test JSON export with null values."""
        t = pxt.create_table('test_json_nulls', {'c_int': pxt.Int, 'c_string': pxt.String})
        t.insert([{'c_int': 1, 'c_string': None}, {'c_int': None, 'c_string': 'hello'}])

        json_path = tmp_path / 'nulls.json'
        pxt.io.export_json(t, json_path)

        with open(json_path, encoding='utf-8') as f:
            exported = json.load(f)
        assert len(exported) == 2
        assert exported[0]['c_string'] is None
        assert exported[1]['c_int'] is None
