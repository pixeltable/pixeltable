from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import call, patch

import pytest
import requests

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.env import Env
from pixeltable.service.pxtfs_protocol import GetBucketCredentialsResponse
from pixeltable.utils.http import fetch_url
from pixeltable.utils.object_stores import ObjectOps, ObjectPath, StorageTarget

from .utils import (
    CLOUD_DB_ROOT_URIS,
    cloud_env_configured,
    home_bucket_uri,
    pxt_raises,
    skip_test_if_no_pxt_credentials,
    skip_test_if_not_installed,
    validate_update_status,
)

pytestmark = pytest.mark.db_roots('local', reason='exercises ObjectOps/object-store internals')


def _pxt_dest_uri() -> str:
    """The pytest prefix in the home bucket of the 'cloud' root's database."""
    if not cloud_env_configured():
        pytest.skip('the cloud environment is unconfigured')
    return f'{home_bucket_uri(CLOUD_DB_ROOT_URIS["cloud"])}/pytest'


def _bucket_credentials(no_space_left: bool = False) -> GetBucketCredentialsResponse:
    """Stand-in credentials for patching get_bucket_credentials."""
    return GetBucketCredentialsResponse(
        access_key_id='key',
        secret_access_key='secret',
        session_token='token',
        endpoint_url='https://r2.example.com',
        resolved_bucket_name='physical-home',
        ttl_seconds=3600,
        no_space_left=no_space_left,
    )


class TestPxtStore:
    """Tests for Pixeltable-managed storage (pxtfs:// home buckets)."""

    def test_insert_and_select(self, uses_db: None) -> None:
        """Insert a local file with a pxtfs:// destination, then verify it can be read back."""
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()

        dest_uri = f'{_pxt_dest_uri()}/bucket1'

        t = pxt.create_table('test_pxt_store', schema={'img': pxt.Image | None})
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

        dest_uri = f'{_pxt_dest_uri()}/src'

        src_table = pxt.create_table('pxt_src', schema={'img': pxt.Image | None})
        src_table.add_computed_column(img_stored=src_table.img.rotate(90), destination=dest_uri)
        validate_update_status(
            src_table.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]), expected_rows=1
        )

        pxt_url = src_table.select(src_table.img_stored.fileurl).collect()['img_stored_fileurl'][0]
        assert pxt_url.startswith('pxtfs://')

        reader_table = pxt.create_table('pxt_reader', schema={'img': pxt.Image | None})
        validate_update_status(reader_table.insert([{'img': pxt_url}]), expected_rows=1)

        result = reader_table.collect()
        assert len(result) == 1
        assert result['img'][0] is not None

    def test_delete_on_drop(self, uses_db: None) -> None:
        """Verify objects in pxt store are cleaned up when the table is dropped."""
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()

        dest_uri = f'{_pxt_dest_uri()}/drop_test'

        t = pxt.create_table('test_pxt_drop', schema={'img': pxt.Image | None})
        t.add_computed_column(img_rot=t.img.rotate(90), destination=dest_uri)
        validate_update_status(
            t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]), expected_rows=1
        )
        assert ObjectOps.count(t._id, dest=dest_uri) == 1

        save_id = t._id
        pxt.drop_table(t)
        assert ObjectOps.count(save_id, dest=dest_uri) == 0

    def test_presigned_url(self, uses_db: None) -> None:
        """A presigned URL for a home-bucket object serves the object, and a byte range of it, which a video player
        asks for to seek."""
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()

        dest_uri = f'{_pxt_dest_uri()}/presigned'
        t = pxt.create_table('test_pxt_presigned', schema={'img': pxt.Image | None})
        t.add_computed_column(img_rot=t.img.rotate(90), destination=dest_uri)
        validate_update_status(
            t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}]), expected_rows=1
        )
        url = t.select(t.img_rot.fileurl).collect()['img_rot_fileurl'][0]

        signed_url = ObjectOps.presigned_url(url, expiration_seconds=300)
        whole = requests.get(signed_url, timeout=30)
        assert whole.status_code == 200
        assert len(whole.content) > 5
        part = requests.get(signed_url, headers={'Range': 'bytes=0-4'}, timeout=30)
        assert part.status_code == 206
        assert part.content == whole.content[:5]
        pxt.drop_table(t)

    def test_no_space_left(self, uses_db: None) -> None:
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()
        from pixeltable.utils import pxt_store

        dest_uri = f'{_pxt_dest_uri()}/quota_test'
        t = pxt.create_table('test_pxt_quota', schema={'img': pxt.Image | None})
        t.add_computed_column(img_rot=t.img.rotate(90), destination=dest_uri)

        img = 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'
        validate_update_status(t.insert([{'img': img}]), expected_rows=1)

        soa = ObjectPath.parse_object_storage_addr(dest_uri, allow_obj_name=False)
        real_entry = pxt_store._get_or_create_pxt_store_entry(soa.account, soa.account_extension, soa.container)
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

    def test_reads_share_credentials(self, init_env: None, tmp_path: Path) -> None:
        """Reading objects from many directories of a home bucket fetches credentials and builds a boto3 session
        once, rather than once per directory: media files are stored in random shard directories."""
        skip_test_if_not_installed('boto3')
        from pixeltable.utils import pxt_store
        from pixeltable.utils.s3_store import S3Store

        # a database no other test has used, so its entry is not cached yet
        db = f'db_{uuid.uuid4().hex}'
        home = f'pxtfs://org1:{db}/home'
        with (
            patch.object(pxt_store, 'get_bucket_credentials', return_value=_bucket_credentials()) as get_credentials,
            patch.object(S3Store, 'copy_object_to_local_file') as download,
        ):
            store = ObjectOps.get_store(home, False)
            tbl_id = uuid.uuid4()
            urls = [store.resolve_destination(tbl_id, 0, 1, ext='.jpg').url for _ in range(100)]
            urls.append(f'{home}/uploads/{uuid.uuid4().hex}/0.jpg')
            for url in urls:
                ObjectOps.copy_object_to_local_file(url, tmp_path / 'obj')

        assert len({url.rsplit('/', 1)[0] for url in urls}) > 90
        assert download.call_count == len(urls)
        get_credentials.assert_called_once_with('org1', db, 'home', None)

    def test_pxt_buckets_destination(self, init_env: None) -> None:
        """A destination spelled pxt://org:db/buckets/home/... is the store its pxtfs:// spelling names: a new file
        gets the same object key and the same stored URL."""
        skip_test_if_not_installed('boto3')
        from pixeltable.utils import object_stores, pxt_store

        db = f'db_{uuid.uuid4().hex}'
        tbl_id, file_id = uuid.uuid4(), uuid.uuid4()
        dests = []
        with patch.object(pxt_store, 'get_bucket_credentials', return_value=_bucket_credentials()) as get_credentials:
            for dest_uri in (f'pxtfs://org1:{db}/home/media', f'pxt://org1:{db}/buckets/home/media'):
                store = ObjectOps.get_store(dest_uri, False)
                # the file name is random; fix it, so the two destinations can be compared whole
                with patch.object(object_stores.uuid, 'uuid4', return_value=file_id):
                    dests.append(store.resolve_destination(tbl_id, 0, 1, ext='.jpg'))

        assert dests[1] == dests[0]
        assert dests[1].url == f'pxtfs://org1:{db}/home/{dests[1].remote_key}'
        assert dests[1].remote_key.startswith(f'media/{tbl_id.hex}/')
        # both spellings share one cached session
        get_credentials.assert_called_once_with('org1', db, 'home', None)

    def test_fetch_url_keeps_bucket_credentials(self, init_env: None) -> None:
        """fetch_url(), which pods call, still reads a home bucket, in either spelling, with credentials for the bucket:
        a pod's reads take no control-plane call per file."""
        skip_test_if_not_installed('boto3')
        from pixeltable.utils import pxt_store
        from pixeltable.utils.s3_store import S3Store

        db = f'db_{uuid.uuid4().hex}'
        with (
            patch.object(pxt_store, 'get_bucket_credentials', return_value=_bucket_credentials()) as get_credentials,
            patch.object(S3Store, 'copy_object_to_local_file') as download,
        ):
            fetch_url(f'pxtfs://org1:{db}/home/k/1.jpg')
            fetch_url(f'pxt://org1:{db}/buckets/home/k/2.jpg')

        get_credentials.assert_called_once_with('org1', db, 'home', None)
        assert [c.args[0] for c in download.call_args_list] == ['1.jpg', '2.jpg']

    def test_quota_recheck(self, init_env: None, tmp_path: Path) -> None:
        """A write rejected for lack of space checks the quota again at most once per interval, and keeps the cached
        state if the check fails; once space is freed, the next check lets writes through."""
        skip_test_if_not_installed('boto3')
        from pixeltable.utils import pxt_store
        from pixeltable.utils.s3_store import S3Store

        home = f'pxtfs://org1:db_{uuid.uuid4().hex}/home'
        src = tmp_path / 'obj.jpg'
        src.write_bytes(b'data')
        full = _bucket_credentials(no_space_left=True)
        with (
            patch.object(pxt_store, 'get_bucket_credentials', return_value=full) as get_credentials,
            patch.object(S3Store, 'copy_local_file', side_effect=lambda src_path, dest: dest.url) as upload,
        ):
            with pytest.warns(excs.PixeltableWarning, match='has no space left'):
                store = ObjectOps.get_store(home, False)
            assert isinstance(store, pxt_store.PxtStore)
            entry = store._pxt_store_entry
            dest = store.resolve_destination(uuid.uuid4(), 0, 1, ext='.jpg')

            def write() -> str:
                # each write's store reuses the cached entry, as a later request's would
                return ObjectOps.get_store(home, False).copy_local_file(src, dest)

            def let_interval_pass() -> None:
                entry.quota_checked_at -= pxt_store._QUOTA_RECHECK_INTERVAL_S

            # within the interval, the cached state rejects the write without a check
            with pxt_raises(excs.ErrorCode.STORE_UNAVAILABLE, match='No space left'):
                write()
            assert get_credentials.call_count == 1

            # a failed check keeps the cached state and starts a new interval
            let_interval_pass()
            get_credentials.side_effect = excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR, 'unreachable', provider='pixeltable_cloud'
            )
            for _ in range(2):
                with pxt_raises(excs.ErrorCode.STORE_UNAVAILABLE, match='No space left'):
                    write()
            assert get_credentials.call_count == 2

            # once space is freed, the next check lets the write through
            let_interval_pass()
            get_credentials.side_effect = None
            get_credentials.return_value = _bucket_credentials()
            assert write() == dest.url
            assert get_credentials.call_count == 3

        upload.assert_called_once()

    def test_scoped_credentials(self, init_env: None) -> None:
        """A store with scope_credentials fetches credentials for its prefix only, in a session that is not cached:
        an upload sink's prefix belongs to one request and is never reused."""
        skip_test_if_not_installed('boto3')
        from pixeltable.utils import pxt_store

        db = f'db_{uuid.uuid4().hex}'
        prefixes = [f'uploads/{uuid.uuid4().hex}/' for _ in range(3)]
        with patch.object(pxt_store, 'get_bucket_credentials', return_value=_bucket_credentials()) as get_credentials:
            for prefix in prefixes:
                ObjectOps.get_store(f'pxtfs://org1:{db}/home/{prefix}', False, scope_credentials=True)

        assert get_credentials.call_args_list == [call('org1', db, 'home', prefix) for prefix in prefixes]
        cached = Env.get().object_store_clients(StorageTarget.PIXELTABLE_STORE).clients
        assert not any(db in key for key in cached)

    def test_same_prefix_shares_credentials(self, uses_db: None) -> None:
        """Verify that two columns with the same pxtfs:// destination share a single cached credential entry."""
        skip_test_if_not_installed('boto3')
        skip_test_if_no_pxt_credentials()
        from pixeltable.utils.pxt_store import PxtStore
        from pixeltable.utils.s3_store import S3Store

        soa = ObjectPath.parse_object_storage_addr(f'{_pxt_dest_uri()}/shared', allow_obj_name=False)

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

        soa = ObjectPath.parse_object_storage_addr(f'{_pxt_dest_uri()}/refresh_test', allow_obj_name=False)
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
