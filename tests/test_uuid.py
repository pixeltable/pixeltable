"""
Tests for UUID type and UUID primary key functionality.
"""

import uuid

import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from tests.utils import ReloadTester, validate_update_status


class TestUUID:
    @pytest.mark.parametrize('uuid_fn, uuid_version', [(pxtf.uuid.uuid4, 4), (pxtf.uuid.uuid7, 7)])
    def test_uuid_function(self, uuid_fn: pxt.Function, uuid_version: int, reset_db: None) -> None:
        t = pxt.create_table('test_uuid_tbl', {'id': pxt.Int})
        validate_update_status(t.insert([{'id': 1}, {'id': 2}, {'id': 3}]), expected_rows=3)

        res = t.select(uuid_col=uuid_fn()).collect()
        assert len(res) == 3
        assert all(isinstance(u, uuid.UUID) for u in res['uuid_col'])
        assert all(u.version == uuid_version for u in res['uuid_col'])
        # Verify all UUIDs are unique
        assert len(set(res['uuid_col'])) == 3

        t.add_computed_column(uuid_col=uuid_fn())
        res = t.select(t.id, t.uuid_col).collect()
        assert len(res) == 3
        assert all(isinstance(u, uuid.UUID) for u in res['uuid_col'])
        assert all(u.version == uuid_version for u in res['uuid_col'])
        # Verify all UUIDs are unique
        assert len(set(res['uuid_col'])) == 3

    def test_uuid_type(self, reset_db: None, reload_tester: ReloadTester) -> None:
        # Test UUIDs of different versions
        test_uuids: list[uuid.UUID] = [
            uuid.uuid1(),
            uuid.uuid3(uuid.NAMESPACE_DNS, 'pixeltable.com'),  # Version 3 (MD5 hash)
            uuid.uuid4(),
            uuid.uuid5(uuid.NAMESPACE_DNS, 'pixeltable.com'),  # Version 5 (SHA-1 hash)
        ]

        # Test basic UUID column operations: insert and query
        t = pxt.create_table('test_uuid_tbl', {'uuid_col': pxt.UUID})
        validate_update_status(t.insert({'uuid_col': u} for u in test_uuids), expected_rows=len(test_uuids))

        # Query all UUIDs
        res = reload_tester.run_query(t.select(t.uuid_col))
        assert len(res) == len(test_uuids)
        assert set(res['uuid_col']) == set(test_uuids)

        # Query with where clause
        first_uuid = test_uuids[0]
        res = reload_tester.run_query(t.where(t.uuid_col == first_uuid))
        assert len(res) == 1
        assert res['uuid_col'][0] == first_uuid

        # Test UUID from string conversion
        test_uuid_str = '550e8400-e29b-41d4-a716-446655440000'
        test_uuid = uuid.UUID(test_uuid_str)
        t2 = pxt.create_table('test_uuid_from_string', {'uuid_col': pxt.UUID})
        validate_update_status(t2.insert([{'uuid_col': test_uuid_str}]), expected_rows=1)
        res = t2.select(t2.uuid_col).collect()
        assert res['uuid_col'][0] == test_uuid

        # Test UUID comparison operations
        t3 = pxt.create_table('test_uuid_comparison', {'uuid_col': pxt.UUID})
        validate_update_status(t3.insert({'uuid_col': u} for u in test_uuids), expected_rows=len(test_uuids))

        # Test equality
        first_uuid = test_uuids[0]
        res = reload_tester.run_query(t3.where(t3.uuid_col == first_uuid))
        assert len(res) == 1

        # Test inequality
        res = reload_tester.run_query(t3.where(t3.uuid_col != first_uuid))
        assert len(res) == len(test_uuids) - 1

        # Test IN
        uuids_to_match = [test_uuids[0], test_uuids[1]]
        res = reload_tester.run_query(t3.where(t3.uuid_col.isin(uuids_to_match)))
        assert len(res) == 2
        assert set(res['uuid_col']) == set(uuids_to_match)

        # Test nullable UUID columns
        t4 = pxt.create_table('test_uuid_nullable', {'uuid_col': pxt.UUID})
        validate_update_status(t4.insert([{'uuid_col': None}]), expected_rows=1)
        validate_update_status(t4.insert([{'uuid_col': uuid.uuid4()}]), expected_rows=1)

        res = reload_tester.run_query(t4.select(t4.uuid_col))
        assert res['uuid_col'][0] is None
        assert isinstance(res['uuid_col'][1], uuid.UUID)

        # Test required UUID columns
        t5 = pxt.create_table('test_uuid_required', {'uuid_col': pxt.Required[pxt.UUID]})
        validate_update_status(t5.insert([{'uuid_col': uuid.uuid4()}]), expected_rows=1)

        # Should raise error for None
        with pytest.raises(pxt.Error):
            t5.insert([{'uuid_col': None}])

        # Verify queries work after reload
        reload_tester.run_reload_test()

    def test_uuid_primary_key(self, reset_db: None, reload_tester: ReloadTester) -> None:
        # Test creating a table with a UUID primary key using computed column
        t = pxt.create_table('test_uuid_pk_tbl1', {'id': pxtf.uuid.uuid4(), 'data': pxt.String}, primary_key=['id'])

        # Verify UUID column is created as primary key
        metadata = t.get_metadata()
        assert metadata['columns']['id']['is_primary_key'] is True
        assert metadata['columns']['id']['type_'] == 'Required[UUID]'
        assert metadata['columns']['id']['computed_with'] is not None

        # Insert rows - UUID column should auto-generate UUIDs
        validate_update_status(t.insert([{'data': 'test1'}, {'data': 'test2'}, {'data': 'test3'}]), expected_rows=3)

        # Query and verify UUIDs are generated
        res = reload_tester.run_query(t.select(t.id, t.data))
        assert len(res) == 3
        assert all(isinstance(u, uuid.UUID) for u in res['id'])
        assert len(set(res['id'])) == 3  # All UUIDs should be unique
        assert res['data'] == ['test1', 'test2', 'test3']

        reload_tester.run_reload_test()
