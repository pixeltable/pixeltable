import uuid

import pytest

import pixeltable as pxt
from tests.conftest import clean_db
from tests.utils import assert_resultset_eq, reload_catalog


@pytest.mark.skip(reason='Turned off by default until we are confident that internal-api.pixeltable.com is stable')
class TestReplica:
    def test_replica_round_trip(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'icol': pxt.Int, 'scol': pxt.String})
        t.insert({'icol': i, 'scol': f'string {i}'} for i in range(10))
        snap = pxt.create_snapshot('test_snapshot', t)
        data = snap.head(n=500)

        remote_uri = f'pxt://asiegel/test_{uuid.uuid4().hex}'
        _ = pxt.create_replica(remote_uri, source=snap)

        clean_db()
        reload_catalog()

        replica = pxt.create_replica('test_replica', source=remote_uri)
        replica_data = replica.head(n=500)

        assert_resultset_eq(data, replica_data)
