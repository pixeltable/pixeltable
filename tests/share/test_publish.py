import uuid

import pytest

import pixeltable as pxt
from tests.conftest import clean_db
from tests.utils import assert_resultset_eq, get_image_files, reload_catalog, skip_test_if_no_pxt_credentials


class TestPublish:
    @pytest.mark.parametrize('org_slug', ['pxt-test', 'pxt-test:main', 'pxt-test:my-db'])
    def test_publish_round_trip(self, reset_db: None, org_slug: str) -> None:
        """
        Test a publish/replicate/drop round trip, with three different organization slug configurations:
        - Default main ('pxt-test')
        - Explicit main ('pxt-test:main')
        - Non-main database ('pxt-test:my-db')

        Tests publishing a snapshot and a base table with different versions, containing an image column, to test all
        of the basic data sharing modes.
        """
        skip_test_if_no_pxt_credentials()

        test_imgs = get_image_files()

        tbl = pxt.create_table('tbl', {'icol': pxt.Int, 'scol': pxt.String, 'imgcol': pxt.Image})
        tbl.insert({'icol': i, 'scol': f'string {i}', 'imgcol': test_imgs[i]} for i in range(10))
        snap = pxt.create_snapshot('snap', tbl)
        snap_data = snap.head(n=500)

        tbl.insert({'icol': i, 'scol': f'string {i}', 'imgcol': test_imgs[i]} for i in range(10, 20))
        tbl_data = tbl.head(n=500)

        snap_remote_uri = f'pxt://{org_slug}/test_{uuid.uuid4().hex}'
        tbl_remote_uri = f'pxt://{org_slug}/test_{uuid.uuid4().hex}'
        pxt.publish(snap, snap_remote_uri)
        pxt.publish(tbl, tbl_remote_uri)

        clean_db()
        reload_catalog()

        snap_replica = pxt.replicate(snap_remote_uri, 'snap_replica')
        snap_replica_data = snap_replica.head(n=500)

        tbl_replica = pxt.replicate(tbl_remote_uri, 'tbl_replica')
        tbl_replica_data = tbl_replica.head(n=500)

        pxt.drop_table(snap_remote_uri)
        pxt.drop_table(tbl_remote_uri)

        assert_resultset_eq(snap_data, snap_replica_data, compare_col_names=True)
        assert_resultset_eq(tbl_data, tbl_replica_data, compare_col_names=True)

    def test_remote_tbl_ops_errors(self, reset_db: None) -> None:
        with pytest.raises(pxt.Error, match=r'Cannot use `force=True` with a cloud replica URI.'):
            pxt.drop_table('pxt://pxt-test/test', force=True)
        with pytest.raises(
            pxt.Error, match=r"`destination_uri` must be a remote Pixeltable URI with the prefix 'pxt://'"
        ):
            pxt.publish('tbl', 'not-a-uri')
        with pytest.raises(pxt.Error, match=r"`remote_uri` must be a remote Pixeltable URI with the prefix 'pxt://'"):
            pxt.replicate('not-a-uri', 'replica')
