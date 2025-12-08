import uuid

import pixeltable as pxt
from pixeltable.functions.uuid import uuid4

from ..utils import ReloadTester, validate_update_status


class TestUUID:
    def test_uuid4(self, reset_db: None, reload_tester: ReloadTester) -> None:
        """Test the uuid4() UDF function."""
        t = pxt.create_table('test_uuid_tbl', {'id': pxt.Int})
        validate_update_status(t.insert({'id': i} for i in range(5)), expected_rows=5)

        # Test uuid4 in select
        res = t.select(t.id, new_uuid=uuid4()).collect()
        assert len(res) == 5
        uuids = res['new_uuid']
        assert all(isinstance(u, uuid.UUID) for u in uuids)
        assert len(set(uuids)) == 5  # All should be unique

        # Test uuid4 in computed column
        t.add_computed_column(uuid_col=uuid4())
        res = reload_tester.run_query(t.select(t.id, t.uuid_col))
        assert len(res) == 5
        assert all(isinstance(u, uuid.UUID) for u in res['uuid_col'])
        assert len(set(res['uuid_col'])) == 5  # All should be unique

        reload_tester.run_reload_test()
