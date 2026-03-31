import datetime
import json
import pathlib

import pixeltable as pxt

from ..utils import create_all_datatypes_tbl, get_json_file, validate_update_status


class TestJson:
    def test_export_all_types(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Export a table with every supported type and verify the JSON output."""
        t = create_all_datatypes_tbl()
        rows = t.collect()

        json_path = tmp_path / 'all_types.json'
        pxt.io.export_json(t, json_path, indent=2)

        with open(json_path, encoding='utf-8') as f:
            exported = json.load(f)

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

        json_path = tmp_path / 'nulls.json'
        pxt.io.export_json(t, json_path)

        with open(json_path, encoding='utf-8') as f:
            exported = json.load(f)

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
        json_path = tmp_path / 'filtered.json'
        pxt.io.export_json(t.where(t.c_int < 5), json_path)
        with open(json_path, encoding='utf-8') as f:
            assert len(json.load(f)) == 5

        # Column subset
        json_path2 = tmp_path / 'subset.json'
        pxt.io.export_json(t.select(t.c_string), json_path2)
        with open(json_path2, encoding='utf-8') as f:
            exported = json.load(f)
        assert len(exported) == 10
        assert list(exported[0].keys()) == ['c_string']

    def test_export_indent_and_encoding(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test compact vs pretty output and non-ASCII preservation."""
        source_path = get_json_file('example.json')
        t = pxt.io.import_json('test_json_fmt', source_path)

        # Compact
        compact_path = tmp_path / 'compact.json'
        pxt.io.export_json(t, compact_path)
        compact = compact_path.read_text(encoding='utf-8')
        assert '\n' not in compact.strip() or compact.count('\n') <= 1

        # Pretty-printed with non-ASCII preserved
        pretty_path = tmp_path / 'pretty.json'
        pxt.io.export_json(t, pretty_path, indent=2)
        pretty = pretty_path.read_text(encoding='utf-8')
        assert pretty.count('\n') > 1
        assert 'Manwë' in pretty

    def test_round_trip(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Export JSON, re-import, and verify data matches."""
        source_path = get_json_file('example.json')
        t = pxt.io.import_json('test_json_rt', source_path)

        json_path = tmp_path / 'round_trip.json'
        pxt.io.export_json(t, json_path, indent=2)

        t2 = pxt.io.import_json('test_json_rt2', str(json_path))

        original = t.order_by(t.name).collect()
        reimported = t2.order_by(t2.name).collect()

        assert original == reimported

    def test_round_trip_media(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Export JSON with media columns, re-import, and verify file URLs survive the round-trip."""
        t = create_all_datatypes_tbl()

        json_path = tmp_path / 'round_trip_media.json'
        pxt.io.export_json(t, json_path, indent=2)

        # Build schema overrides from the original table, excluding binary (not exportable to JSON)
        schema_overrides = {name: ct for name, ct in t._get_schema().items() if not ct.is_binary_type()}
        t2 = pxt.io.import_json('test_json_rt_media', str(json_path), schema_overrides=schema_overrides)

        # Select only columns that survive JSON export (binary is excluded)
        exportable_cols = [getattr(t, name) for name in t2.columns()]
        original = t.select(*exportable_cols).order_by(t.row_id).collect()
        reimported = t2.order_by(t2.row_id).collect()

        assert original == reimported
