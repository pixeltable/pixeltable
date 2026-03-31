from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import pixeltable as pxt
from pixeltable.utils import pxt_store
from pixeltable.utils.object_stores import ObjectOps

from .utils import skip_test_if_no_pxt_credentials, skip_test_if_not_installed, validate_update_status

PXT_DEST_URI = 'pxt://pixeltable:main/home/pytest'


@pytest.mark.skip('Skip tests until pxt store changes are in the cloud')
class TestPxtStore:
    """Tests for Pixeltable-managed storage (pxt:// home buckets)."""

    def test_insert_and_select(self, uses_db: None) -> None:
        """Insert a local file with a pxt:// destination, then verify it can be read back."""
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()
        dest_uri = f'{PXT_DEST_URI}/bucket1'

        t = pxt.create_table('test_pxt_store', schema={'img': pxt.Image})
        t.add_computed_column(img_rot=t.img.rotate(90), destination=dest_uri)
        validate_update_status(
            t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]), expected_rows=1
        )

        result = t.select(t.img_rot.fileurl).collect()
        assert len(result) == 1
        file_url = result['img_rot_fileurl'][0]
        assert file_url.startswith('pxt://'), f'Expected pxt:// URL, got: {file_url}'
        assert ObjectOps.count(t._id, dest=dest_uri) == 1

    def test_select_from_pxt_url(self, uses_db: None) -> None:
        """Upload a file to the pxt store, then insert its pxt:// URL into a new table and read it."""
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()
        dest_uri = f'{PXT_DEST_URI}/src'

        src_table = pxt.create_table('pxt_src', schema={'img': pxt.Image})
        src_table.add_computed_column(img_stored=src_table.img.rotate(90), destination=dest_uri)
        validate_update_status(
            src_table.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]), expected_rows=1
        )

        pxt_url = src_table.select(src_table.img_stored.fileurl).collect()['img_stored_fileurl'][0]
        assert pxt_url.startswith('pxt://')

        reader_table = pxt.create_table('pxt_reader', schema={'img': pxt.Image})
        validate_update_status(reader_table.insert([{'img': pxt_url}]), expected_rows=1)

        result = reader_table.collect()
        assert len(result) == 1
        assert result['img'][0] is not None

    def test_delete_on_drop(self, uses_db: None) -> None:
        """Verify objects in pxt store are cleaned up when the table is dropped."""
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()
        dest_uri = f'{PXT_DEST_URI}/drop_test'

        t = pxt.create_table('test_pxt_drop', schema={'img': pxt.Image})
        t.add_computed_column(img_rot=t.img.rotate(90), destination=dest_uri)
        validate_update_status(
            t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]), expected_rows=1
        )
        assert ObjectOps.count(t._id, dest=dest_uri) == 1

        save_id = t._id
        pxt.drop_table(t)
        assert ObjectOps.count(save_id, dest=dest_uri) == 0

    def test_no_space_left(self, uses_db: None) -> None:
        """Verify that writes are blocked and reads/deletes still work when no_space_left is set."""
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()
        dest_uri = f'{PXT_DEST_URI}/quota_test'

        t = pxt.create_table('test_pxt_quota', schema={'img': pxt.Image})
        t.add_computed_column(img_rot=t.img.rotate(90), destination=dest_uri)

        validate_update_status(
            t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]), expected_rows=1
        )
        assert ObjectOps.count(t._id, dest=dest_uri) == 1

        # Simulate quota exhaustion by injecting an entry with no_space_left=True.
        # We patch _get_or_create_pxt_store_entry so every thread (including insert's
        # worker pool) gets the controlled entry regardless of its local cache state.
        quota_entry = MagicMock()
        quota_entry.no_space_left = True
        quota_entry.bucket_name = 'mock-bucket'

        with patch.object(pxt_store, '_get_or_create_pxt_store_entry', return_value=quota_entry):
            with pytest.raises(pxt.Error, match='No space left'):
                t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])

            # Reads go through copy_object_to_local_file (download), not copy_local_file,
            # so they are unaffected by no_space_left.
            result = t.select(t.img_rot.fileurl).collect()
            assert len(result) == 1

        # Outside the patch, the real entry is used again and writes succeed.
        validate_update_status(
            t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]), expected_rows=1
        )
        assert ObjectOps.count(t._id, dest=dest_uri) == 2
