from contextlib import redirect_stdout
from io import StringIO
import uuid

import pytest

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.utils.console_output import ConsoleMessageFilter, ConsoleOutputHandler
from tests.conftest import clean_db
from tests.utils import assert_resultset_eq, capture_console_output, get_image_files, reload_catalog, skip_test_if_no_pxt_credentials


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

        with capture_console_output() as sio:
            tbl_replica_2 = pxt.replicate(tbl_remote_uri, 'tbl_replica')

        assert f'Replica \'tbl_replica\' is already up to date with source: {tbl_remote_uri}' in sio.getvalue()
        assert tbl_replica_2 is tbl_replica

        if ':' in org_slug:
            # Canonical URI; we expect the URIs to match exactly
            assert snap_replica._get_pxt_uri() == snap_remote_uri
            assert tbl_replica._get_pxt_uri() == tbl_remote_uri

        pxt.drop_table(snap_remote_uri)
        pxt.drop_table(tbl_remote_uri)

        assert_resultset_eq(snap_data, snap_replica_data, compare_col_names=True)
        assert_resultset_eq(tbl_data, tbl_replica_data, compare_col_names=True)

    def test_push_pull(self, reset_db: None) -> None:
        skip_test_if_no_pxt_credentials()

        tbl = pxt.create_table('tbl', {'icol': pxt.Int, 'scol': pxt.String})
        remote_uri = f'pxt://pxt-test/test_{uuid.uuid4().hex}'
        pxt.publish(tbl, remote_uri)
        result_sets: list[pxt.dataframe.DataFrameResultSet] = []
        for version in range(1, 8):
            tbl.insert({'icol': i, 'scol': f'string {i}'} for i in range(version * 10, version * 10 + 10))
            result_sets.append(tbl.head(n=500))
            tbl.push()

        clean_db()
        reload_catalog()

        tbl_replica = pxt.replicate(f'{remote_uri}:3', 'tbl_replica')
        assert tbl_replica.get_metadata()['version'] == 3
        assert_resultset_eq(result_sets[2], tbl_replica.head(n=500))

        tbl_replica.pull()
        assert tbl_replica.get_metadata()['version'] == len(result_sets)
        assert_resultset_eq(result_sets[-1], tbl_replica.head(n=500))

        pxt.drop_table(remote_uri)

    def test_remote_tbl_ops_errors(self, reset_db: None) -> None:
        with pytest.raises(pxt.Error, match=r'Cannot use `force=True` with a cloud replica URI.'):
            pxt.drop_table('pxt://pxt-test/test', force=True)
        with pytest.raises(
            pxt.Error, match=r"`destination_uri` must be a remote Pixeltable URI with the prefix 'pxt://'"
        ):
            pxt.publish('tbl', 'not-a-uri')
        with pytest.raises(pxt.Error, match=r"`remote_uri` must be a remote Pixeltable URI with the prefix 'pxt://'"):
            pxt.replicate('not-a-uri', 'replica')
