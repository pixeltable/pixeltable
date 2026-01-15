import uuid

import pytest

import pixeltable as pxt
from tests.conftest import clean_db
from tests.utils import (
    assert_resultset_eq,
    capture_console_output,
    get_image_files,
    reload_catalog,
    skip_test_if_no_pxt_credentials,
)


# Bug(PXT-943): non-latest row versions have non-NULL index column values
@pytest.mark.corrupts_db
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

        assert f"Replica 'tbl_replica' is already up to date with source: {tbl_remote_uri}" in sio.getvalue()
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
        result_sets: list[pxt.ResultSet] = []
        for version in range(1, 8):
            tbl.insert({'icol': i, 'scol': f'string {i}'} for i in range(version * 10, version * 10 + 10))
            result_sets.append(tbl.head(n=500))
            tbl.push()

        clean_db()
        reload_catalog()

        tbl_replica = pxt.replicate(f'{remote_uri}:3', 'tbl_replica')
        assert tbl_replica.get_metadata()['version'] == 3
        assert_resultset_eq(result_sets[2], tbl_replica.head(n=500))

        tbl_replica = pxt.get_table('tbl_replica')  # get live version handle
        assert tbl_replica.get_metadata()['version'] == 3
        assert_resultset_eq(result_sets[2], tbl_replica.head(n=500))

        tbl_replica.pull()  # in-place pull()
        assert tbl_replica.get_metadata()['version'] == len(result_sets)
        assert_resultset_eq(result_sets[-1], tbl_replica.head(n=500))

        pxt.drop_table('tbl_replica')

        # Also try specific version replicate of various sorts in a random order
        for version in (3, 5, 1, 2, 6, 4):
            tbl_replica = pxt.replicate(f'{remote_uri}:{version}', 'tbl_replica')
            assert tbl_replica.get_metadata()['version'] == version
            assert_resultset_eq(result_sets[version - 1], tbl_replica.head(n=500))

        tbl_replica = pxt.get_table('tbl_replica')
        assert tbl_replica.get_metadata()['version'] == 6  # latest version that has been retrieved
        assert_resultset_eq(result_sets[5], tbl_replica.head(n=500))

        tbl_replica.pull()
        assert tbl_replica.get_metadata()['version'] == len(result_sets)
        assert_resultset_eq(result_sets[-1], tbl_replica.head(n=500))

        pxt.drop_table(remote_uri)

    def test_push_pull_errors(self, reset_db: None) -> None:
        skip_test_if_no_pxt_credentials()

        tbl = pxt.create_table('tbl', {'icol': pxt.Int, 'scol': pxt.String})
        remote_uri = f'pxt://pxt-test/test_{uuid.uuid4().hex}'
        for version in range(1, 8):
            tbl.insert({'icol': i, 'scol': f'string {i}'} for i in range(version * 10, version * 10 + 10))

        with pytest.raises(
            pxt.Error,
            match=(
                r"push\(\): Table 'tbl' has not yet been published to Pixeltable Cloud. "
                r'To publish it, use `pxt.publish\(\)` instead.'
            ),
        ):
            tbl.push()

        pxt.publish('tbl', remote_uri)

        with pytest.raises(
            pxt.Error,
            match=r"pull\(\): Table 'tbl' is not a replica of a Pixeltable Cloud table \(nothing to `pull\(\)`\).",
        ):
            tbl.pull()

        tbl_3 = pxt.get_table('tbl:3')
        with pytest.raises(
            pxt.Error,
            match=r'push\(\): Cannot push specific-version table handle \'tbl:3\'\. '
            'To push the latest version instead:',
        ):
            tbl_3.push()

        clean_db()
        reload_catalog()

        tbl_replica = pxt.replicate(f'{remote_uri}:7', 'tbl_replica')
        with pytest.raises(
            pxt.Error,
            match=r'pull\(\): Cannot pull specific-version table handle \'tbl_replica:7\'\. '
            'To pull the latest version instead:',
        ):
            tbl_replica.pull()

        with pytest.raises(
            pxt.Error, match=r"push\(\): Cannot push replica table 'tbl_replica'. \(Did you mean `pull\(\)`\?\)"
        ):
            tbl_replica.push()

        tbl_replica = pxt.get_table('tbl_replica')

        with pytest.raises(
            pxt.Error, match=r"push\(\): Cannot push replica table 'tbl_replica'. \(Did you mean `pull\(\)`\?\)"
        ):
            tbl_replica.push()

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
