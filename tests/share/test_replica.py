import uuid

import pytest

import pixeltable as pxt
from tests.conftest import clean_db
from tests.utils import assert_resultset_eq, get_image_files, reload_catalog, skip_test_if_no_pxt_credentials


class TestReplica:
    @pytest.mark.parametrize('org_slug', ['pxt-test', 'pxt-test:main', 'pxt-test:my-db'])
    def test_replica_round_trip(self, reset_db: None, org_slug: str) -> None:
        """
        Test a publish/clone/drop snapshot round trip, with three different organization slug configurations:
        - Default main ('pxt-test')
        - Explicit main ('pxt-test:main')
        - Non-main database ('pxt-test:my-db')
        """
        skip_test_if_no_pxt_credentials()

        test_imgs = get_image_files()

        t = pxt.create_table('test_tbl', {'icol': pxt.Int, 'scol': pxt.String, 'imgcol': pxt.Image})
        t.insert({'icol': i, 'scol': f'string {i}', 'imgcol': test_imgs[i]} for i in range(10))
        snap = pxt.create_snapshot('test_snapshot', t)
        data = snap.head(n=500)

        remote_uri = f'pxt://{org_slug}/test_{uuid.uuid4().hex}'
        _ = pxt.create_replica(remote_uri, source=snap)

        clean_db()
        reload_catalog()

        replica = pxt.create_replica('test_replica', source=remote_uri)
        replica_data = replica.head(n=500)

        pxt.drop_table(remote_uri)

        assert_resultset_eq(data, replica_data)
