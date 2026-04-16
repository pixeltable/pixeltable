import csv
import datetime
import json
import pathlib

import pandas as pd
import pytest

import pixeltable as pxt

from ..utils import create_all_datatypes_tbl, create_test_tbl, get_csv_file, validate_update_status


class TestCsv:
    def test_export_all_types(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Export a table with every supported type and verify the CSV output."""
        t = create_all_datatypes_tbl()
        rows = t.collect()

        csv_path = tmp_path / 'all_types.csv'
        pxt.io.export_csv(t, csv_path)

        df = pd.read_csv(csv_path)
        exported = df.to_dict(orient='records')

        assert len(exported) == len(rows)

        for exp_row, orig_row in zip(exported, rows):
            assert exp_row['c_string'] == orig_row['c_string']
            assert exp_row['c_int'] == orig_row['c_int']
            assert exp_row['c_float'] == pytest.approx(orig_row['c_float'])
            assert exp_row['c_bool'] == orig_row['c_bool']

            assert isinstance(exp_row['c_timestamp'], str)
            assert isinstance(exp_row['c_date'], str)
            assert datetime.date.fromisoformat(exp_row['c_date']) == orig_row['c_date']

            assert exp_row['c_uuid'] == str(orig_row['c_uuid'])
            assert json.loads(exp_row['c_json']) == orig_row['c_json']
            assert json.loads(exp_row['c_array']) == orig_row['c_array'].tolist()

            for col in ['c_image', 'c_video', 'c_audio', 'c_document']:
                assert isinstance(exp_row[col], str), f'{col} should be a string'
                assert exp_row[col] != '', f'{col} should not be empty'

        # Verify media columns export the authoritative file URL, not a cached path
        media_cols = {'c_image': t.c_image, 'c_video': t.c_video, 'c_audio': t.c_audio, 'c_document': t.c_document}
        fileurls = t.select(*[col.fileurl for col in media_cols.values()]).collect()
        for exp_row, url_row in zip(exported, fileurls):
            for col_name in media_cols:
                assert exp_row[col_name] == url_row[f'{col_name}_fileurl']

    def test_export_round_trip(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Export a table to CSV, re-import, and verify equality."""
        t = create_test_tbl('test_csv_rt')

        csv_path = tmp_path / 'round_trip.csv'
        # Select only columns whose types survive CSV round-trip (string, int, float, bool)
        query = t.select(t.c1, t.c1n, t.c2, t.c3, t.c4)
        pxt.io.export_csv(query, csv_path)

        t2 = pxt.io.import_csv('test_csv_rt_reimported', str(csv_path))

        assert query.collect() == t2.collect()

    def test_export_exact_output(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Verify exported CSV matches an expected file exactly."""
        expected_path = get_csv_file('expected_export.csv')
        t = pxt.io.import_csv('test_csv_exact', expected_path, primary_key='c_int')

        csv_path = tmp_path / 'exact.csv'
        pxt.io.export_csv(t.order_by(t.c_int), csv_path)

        assert csv_path.read_text(encoding='utf-8') == pathlib.Path(expected_path).read_text(encoding='utf-8')

    def test_export_with_nulls(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Nulls become empty strings in CSV."""
        t = pxt.create_table(
            'test_csv_nulls', {'c_int': pxt.Int, 'c_string': pxt.String, 'c_float': pxt.Float, 'c_json': pxt.Json}
        )
        t.insert(
            [
                {'c_int': 1, 'c_string': None, 'c_float': None, 'c_json': None},
                {'c_int': None, 'c_string': 'hello', 'c_float': 1.5, 'c_json': {'a': 1}},
            ]
        )

        csv_path = tmp_path / 'nulls.csv'
        pxt.io.export_csv(t, csv_path)

        with open(csv_path, encoding='utf-8') as f:
            exported = list(csv.DictReader(f))

        assert exported[0]['c_string'] == ''
        assert exported[0]['c_float'] == ''
        assert exported[0]['c_json'] == ''
        assert exported[1]['c_int'] == ''

    def test_export_with_query(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test export with filtering and column selection."""
        t = pxt.create_table('test_csv_query', {'c_int': pxt.Int, 'c_string': pxt.String})
        validate_update_status(t.insert([{'c_int': i, 'c_string': f'row_{i}'} for i in range(10)]), expected_rows=10)

        csv_path = tmp_path / 'filtered.csv'
        pxt.io.export_csv(t.where(t.c_int < 5), csv_path)
        with open(csv_path, encoding='utf-8') as f:
            assert len(list(csv.DictReader(f))) == 5

        csv_path2 = tmp_path / 'subset.csv'
        pxt.io.export_csv(t.select(t.c_string), csv_path2)
        with open(csv_path2, encoding='utf-8') as f:
            exported = list(csv.DictReader(f))
        assert len(exported) == 10
        assert list(exported[0].keys()) == ['c_string']

    @pytest.mark.parametrize('delimiter', ['\t', ';', '|'])
    def test_export_custom_delimiter(self, uses_db: None, tmp_path: pathlib.Path, delimiter: str) -> None:
        """Test CSV export with custom delimiters."""
        t = pxt.create_table('test_csv_delim', {'c_int': pxt.Int, 'c_string': pxt.String})
        t.insert([{'c_int': 1, 'c_string': 'hello'}, {'c_int': 2, 'c_string': 'world'}])

        csv_path = tmp_path / 'delimited.csv'
        pxt.io.export_csv(t, csv_path, delimiter=delimiter)

        with open(csv_path, encoding='utf-8') as f:
            exported = list(csv.DictReader(f, delimiter=delimiter))
        assert len(exported) == 2
        assert int(exported[0]['c_int']) == 1
        assert exported[1]['c_string'] == 'world'

    def test_export_non_serializable_json_errors(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Exporting a JSON column with non-serializable values should raise an error."""
        t = create_all_datatypes_tbl(non_serializable_json=True)
        with pytest.raises(pxt.Error, match='not JSON-serializable'):
            pxt.io.export_csv(t, tmp_path / 'should_fail.csv')
