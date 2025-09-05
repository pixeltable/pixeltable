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

        tbl = pxt.create_table('tbl', {'icol': pxt.Int, 'scol': pxt.String, 'imgcol': pxt.Image})
        tbl.insert({'icol': i, 'scol': f'string {i}', 'imgcol': test_imgs[i]} for i in range(10))
        snap = pxt.create_snapshot('snap', tbl)
        snap_data = snap.head(n=500)

        tbl.insert({'icol': i, 'scol': f'string {i}', 'imgcol': test_imgs[i]} for i in range(10, 20))
        # tbl_data = tbl.head(n=500)

        snap_remote_uri = f'pxt://{org_slug}/test_{uuid.uuid4().hex}'
        # tbl_remote_uri = f'pxt://{org_slug}/test_{uuid.uuid4().hex}'
        _ = pxt.create_replica(snap_remote_uri, source=snap)
        # _ = pxt.create_replica(tbl_remote_uri, source=tbl)

        clean_db()
        reload_catalog()

        snap_replica = pxt.create_replica('snap_replica', source=snap_remote_uri)
        snap_replica_data = snap_replica.head(n=500)

        # tbl_replica = pxt.create_replica('tbl_replica', source=tbl_remote_uri)
        # tbl_replica_data = tbl_replica.head(n=500)

        pxt.drop_table(snap_remote_uri)
        # pxt.drop_table(tbl_remote_uri)

        assert_resultset_eq(snap_data, snap_replica_data, compare_col_names=True)
        # assert_resultset_eq(tbl_data, tbl_replica_data, compare_col_names=True)
