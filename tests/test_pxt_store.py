from __future__ import annotations

import pytest

import pixeltable as pxt
from pixeltable.runtime import get_runtime
from pixeltable.utils.object_stores import ObjectOps, StorageTarget

from .utils import skip_test_if_no_pxt_credentials, skip_test_if_not_installed, validate_update_status

PXT_DEST_URI = 'pxt://pixeltable:main/home/pytest'


class TestPxtStore:
    """Tests for Pixeltable-managed storage (pxt:// home buckets)."""

    @staticmethod
    def _resolve_pxt_destination() -> str:
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()
        try:
            ObjectOps.validate_destination(PXT_DEST_URI)
            return PXT_DEST_URI
        except Exception as exc:
            pytest.skip(f'Pixeltable store not reachable: {exc}')

    def test_insert_and_select(self, uses_db: None) -> None:
        """Insert a local file with a pxt:// destination, then verify it can be read back."""
        dest_uri = self._resolve_pxt_destination()
        dest1_uri = f'{dest_uri}/bucket1'

        t = pxt.create_table('test_pxt_store', schema={'img': pxt.Image})
        t.add_computed_column(img_rot=t.img.rotate(90), destination=dest1_uri)
        validate_update_status(
            t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]),
            expected_rows=1,
        )

        result = t.select(t.img_rot.fileurl).collect()
        assert len(result) == 1
        file_url = result['img_rot'][0]
        assert file_url.startswith('pxt://'), f'Expected pxt:// URL, got: {file_url}'

        assert ObjectOps.count(t._id, dest=dest1_uri) == 1

    def test_select_from_pxt_url(self, uses_db: None) -> None:
        """Upload a file to the pxt store, then insert its pxt:// URL into a new table and read it."""
        dest_uri = self._resolve_pxt_destination()
        dest1_uri = f'{dest_uri}/src'

        src_table = pxt.create_table('pxt_src', schema={'img': pxt.Image})
        src_table.add_computed_column(img_stored=src_table.img.rotate(90), destination=dest1_uri)
        validate_update_status(
            src_table.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]),
            expected_rows=1,
        )

        pxt_url = src_table.select(src_table.img_stored.fileurl).collect()['img_stored'][0]
        assert pxt_url.startswith('pxt://')

        reader_table = pxt.create_table('pxt_reader', schema={'img': pxt.Image})
        validate_update_status(reader_table.insert([{'img': pxt_url}]), expected_rows=1)

        result = reader_table.collect()
        assert len(result) == 1
        assert result['img'][0] is not None

    def test_delete_on_drop(self, uses_db: None) -> None:
        """Verify objects in pxt store are cleaned up when the table is dropped."""
        dest_uri = self._resolve_pxt_destination()
        dest1_uri = f'{dest_uri}/drop_test'

        t = pxt.create_table('test_pxt_drop', schema={'img': pxt.Image})
        t.add_computed_column(img_rot=t.img.rotate(90), destination=dest1_uri)
        validate_update_status(
            t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]),
            expected_rows=1,
        )

        assert ObjectOps.count(t._id, dest=dest1_uri) == 1

        save_id = t._id
        pxt.drop_table(t)
        assert ObjectOps.count(save_id, dest=dest1_uri) == 0

    def test_no_space_left(self, uses_db: None) -> None:
        """Verify that writes are blocked and reads/deletes still work when no_space_left is set."""
        dest_uri = self._resolve_pxt_destination()
        dest1_uri = f'{dest_uri}/quota_test'

        t = pxt.create_table('test_pxt_quota', schema={'img': pxt.Image})
        t.add_computed_column(img_rot=t.img.rotate(90), destination=dest1_uri)

        # First insert succeeds -- populates the cache entry
        validate_update_status(
            t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]),
            expected_rows=1,
        )
        assert ObjectOps.count(t._id, dest=dest1_uri) == 1

        # Flip no_space_left on the cached entry
        cd = get_runtime().get_client('pxt_home')
        entry = cd.clients['pixeltable:main']
        entry.no_space_left = True

        # Writes should fail
        with pytest.raises(pxt.Error, match='No space left'):
            t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])

        # Reads should still work
        result = t.select(t.img_rot.fileurl).collect()
        assert len(result) == 1

        # Restore and verify writes work again
        entry.no_space_left = False
        validate_update_status(
            t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]),
            expected_rows=1,
        )
        assert ObjectOps.count(t._id, dest=dest1_uri) == 2
