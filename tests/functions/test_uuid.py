import uuid
from typing import ClassVar

import pytest

import pixeltable as pxt
from pixeltable.functions.uuid import make_uuid

from ..utils import ReloadTester, validate_update_status


class TestUUID:
    # Test UUIDs of different versions
    TEST_UUIDS: ClassVar[list[uuid.UUID]] = [
        uuid.uuid1(),
        uuid.uuid3(uuid.NAMESPACE_DNS, 'pixeltable.com'),  # Version 3 (MD5 hash)
        uuid.uuid4(),
        uuid.uuid5(uuid.NAMESPACE_DNS, 'pixeltable.com'),  # Version 5 (SHA-1 hash)
    ]

    def test_uuid_insert_and_query(self, reset_db: None, reload_tester: ReloadTester) -> None:
        """Test basic UUID column operations: insert and query."""
        t = pxt.create_table('test_uuid_tbl', {'uuid_col': pxt.UUID})
        validate_update_status(t.insert({'uuid_col': u} for u in self.TEST_UUIDS), expected_rows=len(self.TEST_UUIDS))
        test_uuids = self.TEST_UUIDS

        # Query all UUIDs
        res = reload_tester.run_query(t.select(t.uuid_col))
        assert len(res) == len(test_uuids)
        assert res['uuid_col'] == test_uuids

        # Query with where clause
        first_uuid = test_uuids[0]
        res = reload_tester.run_query(t.where(t.uuid_col == first_uuid))
        assert len(res) == 1
        assert res['uuid_col'][0] == first_uuid

        # Verify queries work after reload
        reload_tester.run_reload_test()

    def test_make_uuid(self, reset_db: None, reload_tester: ReloadTester) -> None:
        t = pxt.create_table('test_uuid_tbl', {'id': pxt.Int})
        validate_update_status(t.insert({'id': i} for i in range(5)), expected_rows=5)

        # Test make_uuid in select
        res = t.select(t.id, new_uuid=make_uuid()).collect()
        assert len(res) == 5
        uuids = res['new_uuid']
        assert all(isinstance(u, uuid.UUID) for u in uuids)
        assert len(set(uuids)) == 5  # All should be unique

        # Test make_uuid in computed column
        t.add_computed_column(uuid_col=make_uuid())
        res = reload_tester.run_query(t.select(t.id, t.uuid_col))
        assert len(res) == 5
        assert all(isinstance(u, uuid.UUID) for u in res['uuid_col'])
        assert len(set(res['uuid_col'])) == 5  # All should be unique

        reload_tester.run_reload_test()

    def test_uuid_from_string(self, reset_db: None) -> None:
        t = pxt.create_table('test_uuid_tbl', {'uuid_col': pxt.UUID})
        test_uuid_str = '550e8400-e29b-41d4-a716-446655440000'
        test_uuid = uuid.UUID(test_uuid_str)

        validate_update_status(t.insert([{'uuid_col': test_uuid_str}]), expected_rows=1)
        res = t.select(t.uuid_col).collect()
        assert res['uuid_col'][0] == test_uuid

    def test_uuid_comparison(self, reset_db: None, reload_tester: ReloadTester) -> None:
        t = pxt.create_table('test_uuid_tbl', {'uuid_col': pxt.UUID})
        validate_update_status(t.insert({'uuid_col': u} for u in self.TEST_UUIDS), expected_rows=len(self.TEST_UUIDS))

        # Test equality
        first_uuid = self.TEST_UUIDS[0]
        res = reload_tester.run_query(t.where(t.uuid_col == first_uuid))
        assert len(res) == 1

        # Test inequality
        res = reload_tester.run_query(t.where(t.uuid_col != first_uuid))
        assert len(res) == len(self.TEST_UUIDS) - 1

        # Test IN
        uuids_to_match = [self.TEST_UUIDS[0], self.TEST_UUIDS[1]]
        res = reload_tester.run_query(t.where(t.uuid_col.isin(uuids_to_match)))
        assert len(res) == 2
        assert set(res['uuid_col']) == set(uuids_to_match)

        reload_tester.run_reload_test()

    def test_uuid_nullable(self, reset_db: None, reload_tester: ReloadTester) -> None:
        t = pxt.create_table('test_uuid_tbl', {'uuid_col': pxt.UUID})
        validate_update_status(t.insert([{'uuid_col': None}]), expected_rows=1)
        validate_update_status(t.insert([{'uuid_col': uuid.uuid4()}]), expected_rows=1)

        res = reload_tester.run_query(t.select(t.uuid_col))
        assert res['uuid_col'][0] is None
        assert isinstance(res['uuid_col'][1], uuid.UUID)

        reload_tester.run_reload_test()

    def test_uuid_required(self, reset_db: None) -> None:
        t = pxt.create_table('test_uuid_tbl', {'uuid_col': pxt.Required[pxt.UUID]})
        validate_update_status(t.insert([{'uuid_col': uuid.uuid4()}]), expected_rows=1)

        # Should raise error for None
        with pytest.raises(pxt.Error):
            t.insert([{'uuid_col': None}])
