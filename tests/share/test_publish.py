import uuid
from unittest.mock import MagicMock, patch

import pytest

import pixeltable as pxt
from tests.conftest import clean_db
from tests.utils import assert_resultset_eq, get_image_files, reload_catalog, skip_test_if_no_pxt_credentials


class TestPublish:
    @pytest.mark.parametrize('org_slug', ['pxt-test', 'pxt-test:main', 'pxt-test:my-db'])
    def test_publish_round_trip(self, reset_db: None, org_slug: str) -> None:
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
        pxt.publish(snap, snap_remote_uri)
        # _ = pxt.create_replica(tbl_remote_uri, source=tbl)

        clean_db()
        reload_catalog()

        snap_replica = pxt.replicate(snap_remote_uri, 'snap_replica')
        snap_replica_data = snap_replica.head(n=500)

        # tbl_replica = pxt.create_replica('tbl_replica', source=tbl_remote_uri)
        # tbl_replica_data = tbl_replica.head(n=500)

        pxt.drop_table(snap_remote_uri)
        # pxt.drop_table(tbl_remote_uri)

        assert_resultset_eq(snap_data, snap_replica_data, compare_col_names=True)
        # assert_resultset_eq(tbl_data, tbl_replica_data, compare_col_names=True)

    def test_remote_tbl_ops_errors(self, reset_db: None) -> None:
        with pytest.raises(pxt.Error, match=r'Cannot use `force=True` with a cloud replica URI.'):
            pxt.drop_table('pxt://pxt-test/test', force=True)
        with pytest.raises(
            pxt.Error, match=r"`destination_uri` must be a remote Pixeltable URI with the prefix 'pxt://'"
        ):
            pxt.publish('tbl', 'not-a-uri')
        with pytest.raises(pxt.Error, match=r"`remote_uri` must be a remote Pixeltable URI with the prefix 'pxt://'"):
            pxt.replicate('not-a-uri', 'replica')

    @patch('pixeltable.share.publish.requests.post')
    def test_table_already_replicated_error_message(self, mock_post: MagicMock, reset_db: None) -> None:
        """Test that the 'table already replicated' error provides actionable guidance."""
        # Create a table to publish
        tbl = pxt.create_table('test_tbl', {'col': pxt.Int})
        tbl.insert([{'col': 1}])
        snap = pxt.create_snapshot('test_snap', tbl)

        # Mock the initial publish request to succeed
        mock_initial_response = MagicMock()
        mock_initial_response.status_code = 200
        mock_initial_response.json.return_value = {
            'upload_id': 'test-upload-id',
            'destination_uri': 'https://example.com/bundle.tar.gz'
        }

        # Mock the finalize request to return "already replicated" error
        mock_finalize_response = MagicMock()
        mock_finalize_response.status_code = 400
        mock_finalize_response.text = "That table has already been replicated as 'existing_db.existing_table'."

        # Configure mock to return different responses for different calls
        mock_post.side_effect = [mock_initial_response, mock_finalize_response]

        # Mock the upload function to do nothing
        with patch('pixeltable.share.publish._upload_to_presigned_url'):
            # Verify the error message contains all expected elements
            with pytest.raises(pxt.Error) as exc_info:
                pxt.publish(snap, 'pxt://test-org/test-table')

            error_message = str(exc_info.value)
            # Verify the error message contains:
            # 1. The destination URI
            assert 'pxt://test-org/test-table' in error_message
            # 2. The existing table name (extracted via regex)
            assert 'existing_db.existing_table' in error_message
            # 3. Actionable guidance with drop_table command
            assert 'pxt.drop_table' in error_message
            # 4. Actionable guidance with publish command
            assert 'pxt.publish' in error_message
            # 5. The helpful emoji indicator
            assert '💡' in error_message
