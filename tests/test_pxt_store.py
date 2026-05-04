from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.utils.object_stores import ObjectOps, ObjectPath

from .utils import pxt_raises, skip_test_if_no_pxt_credentials, skip_test_if_not_installed, validate_update_status

PXT_DEST_URI = 'pxtfs://pixeltable:main/home/pytest'


@pytest.mark.skip('Skip tests until pxt store changes are in the cloud')
class TestPxtStore:
    """Tests for Pixeltable-managed storage (pxtfs:// home buckets)."""

    def test_insert_and_select(self, uses_db: None) -> None:
        """Insert a local file with a pxtfs:// destination, then verify it can be read back."""
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
        assert file_url.startswith('pxtfs://'), f'Expected pxtfs:// URL, got: {file_url}'
        assert ObjectOps.count(t._id, dest=dest_uri) == 1

    def test_select_from_pxt_url(self, uses_db: None) -> None:
        """Upload a file to the pxt store, then insert its pxtfs:// URL into a new table and read it."""
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()

        dest_uri = f'{PXT_DEST_URI}/src'

        src_table = pxt.create_table('pxt_src', schema={'img': pxt.Image})
        src_table.add_computed_column(img_stored=src_table.img.rotate(90), destination=dest_uri)
        validate_update_status(
            src_table.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]), expected_rows=1
        )

        pxt_url = src_table.select(src_table.img_stored.fileurl).collect()['img_stored_fileurl'][0]
        assert pxt_url.startswith('pxtfs://')

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
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()
        from pixeltable.utils import pxt_store

        dest_uri = f'{PXT_DEST_URI}/quota_test'
        t = pxt.create_table('test_pxt_quota', schema={'img': pxt.Image})
        t.add_computed_column(img_rot=t.img.rotate(90), destination=dest_uri)

        img = 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'
        validate_update_status(t.insert([{'img': img}]), expected_rows=1)

        soa = ObjectPath.parse_object_storage_addr(dest_uri, allow_obj_name=False)
        real_entry = pxt_store._get_or_create_pxt_store_entry(
            soa.account, soa.account_extension, soa.container, soa.prefix
        )
        quota_entry = pxt_store._PxtStoreCacheEntry(
            client=real_entry.client,
            resource=real_entry.resource,
            physical_bucket_name=real_entry.physical_bucket_name,
            endpoint_url=real_entry.endpoint_url,
            storage_provider=real_entry.storage_provider,
            no_space_left=True,
        )

        with patch.object(pxt_store, '_get_or_create_pxt_store_entry', return_value=quota_entry):
            with pxt_raises(excs.ErrorCode.STORE_UNAVAILABLE, match='No space left'):
                t.insert([{'img': img}])

            result = t.select(t.img_rot.fileurl).collect()
            assert len(result) == 1

        validate_update_status(t.insert([{'img': img}]), expected_rows=1)
        assert ObjectOps.count(t._id, dest=dest_uri) == 2

    def test_separate_prefixes_get_separate_credentials(self, uses_db: None) -> None:
        """Verify that two columns with different prefixes under the same org:db get separate credentials."""
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()
        from pixeltable.utils.pxt_store import PxtStore
        from pixeltable.utils.s3_store import S3Store

        soa1 = ObjectPath.parse_object_storage_addr(f'{PXT_DEST_URI}/dir1', allow_obj_name=False)
        soa2 = ObjectPath.parse_object_storage_addr(f'{PXT_DEST_URI}/dir2', allow_obj_name=False)

        store1 = PxtStore(soa1)
        store2 = PxtStore(soa2)
        assert store1._pxt_store_entry is not store2._pxt_store_entry
        assert isinstance(store1._store, S3Store)
        assert isinstance(store2._store, S3Store)
        assert store1._store.client() is not store2._store.client()

    def test_same_prefix_shares_credentials(self, uses_db: None) -> None:
        """Verify that two columns with the same pxtfs:// destination share a single cached credential entry."""
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()
        from pixeltable.utils.pxt_store import PxtStore
        from pixeltable.utils.s3_store import S3Store

        soa = ObjectPath.parse_object_storage_addr(f'{PXT_DEST_URI}/shared', allow_obj_name=False)

        store1 = PxtStore(soa)
        store2 = PxtStore(soa)
        assert store1._pxt_store_entry is store2._pxt_store_entry
        assert isinstance(store1._store, S3Store)
        assert isinstance(store2._store, S3Store)
        assert store1._store.client() is store2._store.client()

    def test_credentials_refresh(self, uses_db: None) -> None:
        """Verify that botocore automatically refreshes credentials when they expire."""
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()
        from pixeltable.utils.pxt_store import PxtStore

        soa = ObjectPath.parse_object_storage_addr(f'{PXT_DEST_URI}/refresh_test', allow_obj_name=False)
        store = PxtStore(soa)
        refreshable_creds = store._store.client()._get_credentials()  # type: ignore[attr-defined]
        initial_access_key = refreshable_creds.access_key
        initial_token = refreshable_creds.token

        # Backdate expiry to force refresh on next API call
        refreshable_creds._expiry_time = datetime.now(tz=timezone.utc) - timedelta(seconds=1)

        # Trigger home bucket access key refresh
        store.list_objects(return_uri=False)

        assert refreshable_creds._expiry_time > datetime.now(tz=timezone.utc), (
            'Expected expiry_time to be in the future after credential refresh'
        )
        assert refreshable_creds.access_key != initial_access_key or refreshable_creds.token != initial_token, (
            'Expected access key or session token to change after credential refresh'
        )
