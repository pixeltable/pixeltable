import uuid

import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf

from ..utils import validate_update_status


class TestUuid:
    def test_to_string(self, uses_db: None) -> None:
        """Test to_string with different formats"""
        t = pxt.create_table('test_tbl', {'id': pxt.UUID})

        # Create some test UUIDs
        test_uuids = [uuid.uuid4() for _ in range(3)]
        validate_update_status(t.insert({'id': u} for u in test_uuids), expected_rows=len(test_uuids))

        t.add_computed_column(id_standard=pxtf.uuid.to_string(t.id))
        t.add_computed_column(id_standard_2=pxtf.uuid.to_string(t.id, format='standard'))
        t.add_computed_column(id_hex=pxtf.uuid.to_string(t.id, format='hex'))
        results = t.collect()
        for row, expected_uuid in zip(results, test_uuids):
            assert row['id_standard'] == str(expected_uuid)
            assert row['id_standard_2'] == str(expected_uuid)
            assert row['id_hex'] == expected_uuid.hex

        # Test invalid format raises error
        with pytest.raises(pxt.Error, match=r"Invalid format: 'invalid'"):
            t.add_computed_column(id_invalid=pxtf.uuid.to_string(t.id, format='invalid'))
