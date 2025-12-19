from __future__ import annotations

import os
from typing import ClassVar

import pytest
import requests

import pixeltable as pxt
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.functions.net import presigned_url
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import ObjectOps, ObjectPath, StorageTarget

from .utils import rerun, skip_test_if_not_installed


class TestDestination:
    TESTED_DESTINATIONS = (
        StorageTarget.AZURE_STORE,
        StorageTarget.B2_STORE,
        StorageTarget.GCS_STORE,
        StorageTarget.LOCAL_STORE,
        StorageTarget.R2_STORE,
        StorageTarget.S3_STORE,
        StorageTarget.TIGRIS_STORE,
    )

    @classmethod
    def resolve_destination_uri(cls, dest_id: StorageTarget, skip_on_failure: bool = True) -> str | None:
        assert dest_id in cls.TESTED_DESTINATIONS
        uri: str
        match dest_id:
            case StorageTarget.AZURE_STORE:
                uri = 'https://pixeltable1.blob.core.windows.net/pytest'
            case StorageTarget.B2_STORE:
                uri = 'https://s3.us-east-005.backblazeb2.com/pixeltable/pytest'
            case StorageTarget.GCS_STORE:
                if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
                    if skip_on_failure:
                        pytest.skip('GOOGLE_APPLICATION_CREDENTIALS is not set')
                    return None
                uri = 'gs://pxt-gs-test/pytest'
            case StorageTarget.LOCAL_STORE:
                base_path = Config.get().home / 'test-dest'
                for i in range(1, 5):
                    (base_path / f'bucket{i}').mkdir(parents=True, exist_ok=True)
                uri = base_path.as_uri()
            case StorageTarget.S3_STORE:
                uri = 's3://pxt-test/pytest'
            case StorageTarget.R2_STORE:
                uri = 'https://ae60fad96d33636287c3b2e76b88241f.r2.cloudflarestorage.com/pxt-test/pytest'
            case StorageTarget.TIGRIS_STORE:
                uri = 'https://t3.storage.dev/pxt-test/pytest'

        try:
            ObjectOps.validate_destination(uri)
            return uri
        except Exception as exc:
            if skip_on_failure:
                pytest.skip(f'Destination {str(dest_id)!r} not reachable or not configured properly: {exc}')
            return None

    def test_dest_errors(self, reset_db: None) -> None:
        t = pxt.create_table('test_dest_errors', schema={'img': pxt.Image})
        valid_dest = 'tests/data/'

        # Basic tests of the destination parameter: types and store / computed
        with pytest.raises(pxt.Error, match='must be a string or path'):
            t.add_computed_column(img_rot=t.img.rotate(90), destination=27)

        with pytest.raises(pxt.Error, match='only applies to stored computed columns'):
            t.add_computed_column(img_rot=t.img.rotate(90), stored=False, destination=valid_dest)

        with pytest.raises(pxt.Error, match='only applies to stored computed columns'):
            _ = pxt.create_table('test_dest_bad', schema={'img': {'type': pxt.Image, 'destination': f'{valid_dest}'}})

        # Test destination with a non-existent directory
        with pytest.raises(pxt.Error, match='does not exist'):
            t.add_computed_column(img_rot=t.img.rotate(90), destination='non_existent_dir/img_rot')

        # Test destination with a file path instead of a directory
        with pytest.raises(pxt.Error, match='must be a directory, not a file'):
            t.add_computed_column(
                img_rot=t.img.rotate(90), destination='tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'
            )

        # Test with invalid scheme
        with pytest.raises(pxt.Error, match='must be a valid reference to a supported'):
            t.add_computed_column(img_rot=t.img.rotate(90), destination='https://anything/')

    def test_invalid_bucket(self, reset_db: None) -> None:
        skip_test_if_not_installed('boto3')
        t = pxt.create_table('test_invalid_dest', schema={'img': pxt.Image})

        with pytest.raises(
            pxt.Error,
            match="Client error while validating destination for column 'img_rot': "
            "Bucket 'pxt-test-not-a-bucket' not found",
        ):
            t.add_computed_column(img_rot=t.img.rotate(90), destination='s3://pxt-test-not-a-bucket/pytest')

        # The error message on this next one appears to vary by environment.
        msg1 = (
            r'Connection error while validating destination '
            r"'https://a711169187abcf395c01dca4390ee0ea.r2.cloudflarestorage.com/pxt-test/pytest/' "
            r"for column 'img_rot':"
        )
        msg2 = (
            r"Client error while validating destination for column 'img_rot': "
            r"Access denied to bucket 'pxt-test': Forbidden"
        )
        with pytest.raises(pxt.Error, match=f'{msg1}|{msg2}'):
            t.add_computed_column(
                img_rot=t.img.rotate(90),
                destination='https://a711169187abcf395c01dca4390ee0ea.r2.cloudflarestorage.com/pxt-test/pytest',
            )

    def test_dest_parser(self, reset_db: None) -> None:
        a_name = 'acct-name'
        o_name = 'obj-name'
        p_name1 = 'path-name'
        p_name2 = 'path-name/path2-name'
        for s in (
            's3://container',
            f'wasb://container@{a_name}.blob.core.windows.net',
            f'https://{a_name}.blob.core.windows.net/container',
            f'https://{a_name}.r2.cloudflarestorage.com/container',
            'https://s3.us-east-005.backblazeb2.com/container',
            'https://t3.storage.dev/container',
            'https://raw.github.com',
            'file://dir1/dir2/dir3',
            'dir1/dir2/dir3',
        ):
            for allow_obj_name in (False, True):
                ObjectPath.parse_object_storage_addr(s, allow_obj_name)
                ObjectPath.parse_object_storage_addr(s + '/', allow_obj_name)
                ObjectPath.parse_object_storage_addr(s + '/' + p_name1, allow_obj_name)
                ObjectPath.parse_object_storage_addr(s + '/' + p_name1 + '/', allow_obj_name)
                ObjectPath.parse_object_storage_addr(s + '/' + p_name2, allow_obj_name)
                ObjectPath.parse_object_storage_addr(s + '/' + p_name2 + '/', allow_obj_name)
                ObjectPath.parse_object_storage_addr(s + '/' + o_name, allow_obj_name)
                ObjectPath.parse_object_storage_addr(s + '/' + p_name2 + '/' + o_name, allow_obj_name)

        ObjectPath.parse_object_storage_addr(
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000030.jpg',
            allow_obj_name=True,
        )

        ObjectPath.parse_object_storage_addr('file://dir1/dir2/dir3', allow_obj_name=False)
        ObjectPath.parse_object_storage_addr(f'file://dir1/dir2/dir3/{o_name}', allow_obj_name=True)
        ObjectPath.parse_object_storage_addr(f'dir2/dir3/{o_name}', allow_obj_name=True)

    @pytest.mark.parametrize('dest_id', TESTED_DESTINATIONS)
    def test_destination(self, reset_db: None, dest_id: StorageTarget) -> None:
        """Test various media destinations."""
        skip_test_if_not_installed('boto3')
        from pixeltable.utils.s3_store import S3Store

        dest_uri = self.resolve_destination_uri(dest_id)

        dest1_uri = f'{dest_uri}/bucket1'
        dest2_uri = f'{dest_uri}/bucket2'

        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        t.add_computed_column(img_rot1=t.img.rotate(90), destination=None)
        t.add_computed_column(img_rot2=t.img.rotate(180), destination=dest1_uri)
        t.add_computed_column(img_rot3=t.img.rotate(270), destination=dest2_uri)
        print(t.collect())
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        r = t.collect()
        print(r)

        r_dest = t.select(t.img.fileurl, t.img_rot1.fileurl, t.img_rot2.fileurl, t.img_rot3.fileurl).collect()
        print(r_dest)

        print(t.history())

        assert ObjectOps.count(t._id, default_output_dest=True) == 2
        assert ObjectOps.count(t._id, dest=dest1_uri) == 2
        assert ObjectOps.count(t._id, dest=dest2_uri) == 2

        assert ObjectOps.count(t._id, 2, default_output_dest=True) == 1
        assert ObjectOps.count(t._id, 3, dest=dest1_uri) == 1
        assert ObjectOps.count(t._id, 4, dest=dest2_uri) == 1

        assert ObjectOps.count(t._id, 5, default_output_dest=True) == 1
        assert ObjectOps.count(t._id, 5, dest=dest1_uri) == 1
        assert ObjectOps.count(t._id, 5, dest=dest2_uri) == 1

        # Test that we can list objects in the destination
        uris = ObjectOps.list_uris(dest1_uri, n_max=10)
        assert len(uris) >= 2

        # Verify Content-Type is set correctly for S3-compatible stores
        if dest_id in (
            StorageTarget.S3_STORE,
            StorageTarget.R2_STORE,
            StorageTarget.B2_STORE,
            StorageTarget.TIGRIS_STORE,
        ):
            res = t.select(dest1=t.img_rot2.fileurl, dest2=t.img_rot3.fileurl).collect()
            for dest_uri, col_name in ((dest1_uri, 'dest1'), (dest2_uri, 'dest2')):
                store = ObjectOps.get_store(dest_uri, allow_obj_name=False)
                assert isinstance(store, S3Store)
                for d in res[col_name]:
                    addr = ObjectPath.parse_object_storage_addr(d, allow_obj_name=True)
                    content_type = store.get_object_content_type(addr.key)
                    assert content_type == 'image/jpeg', content_type

        # Ensure that all media is removed when the table is dropped
        save_id = t._id
        pxt.drop_table(t)

        assert ObjectOps.count(save_id, default_output_dest=True) == 0
        assert ObjectOps.count(save_id, dest=dest1_uri) == 0
        assert ObjectOps.count(save_id, dest=dest2_uri) == 0

    @pytest.mark.parametrize('dest_id', TESTED_DESTINATIONS)
    def test_dest_two_copies(self, reset_db: None, dest_id: StorageTarget) -> None:
        """Test destination with two Stores receiving copies of the same computed image"""
        dest_uri = self.resolve_destination_uri(dest_id)

        dest1_uri = f'{dest_uri}/bucket1'
        dest2_uri = f'{dest_uri}/bucket2'

        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        t.add_computed_column(img_rot1=t.img.rotate(90), destination=None)
        t.add_computed_column(img_rot2=t.img.rotate(90), destination=dest1_uri)
        t.add_computed_column(img_rot3=t.img.rotate(90), destination=dest2_uri)
        t.add_computed_column(img_rot4=t.img.rotate(90), destination=dest2_uri)  # Try to copy twice to the same dest
        print(t.collect())
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        r = t.collect()
        print(r)

        r_dest = t.select(t.img_rot1.fileurl, t.img_rot2.fileurl, t.img_rot3.fileurl, t.img_rot4.fileurl).collect()
        print(r_dest)

        assert len(r) == 2
        assert len(r) == ObjectOps.count(t._id, default_output_dest=True)
        assert len(r) == ObjectOps.count(t._id, dest=dest1_uri)

        # The outcome of this test is unusual:
        # When the column img_rot4 is ADDED, the computed result for existing rows is not identified
        # as a duplicate, so it is double copied to the destination.
        # When new rows are INSERTED, the results and destinations for img_rot3 and img_rot4 are identified
        # as duplicates, so they are not double copied to the destination.
        assert len(r) + 1 == ObjectOps.count(t._id, dest=dest2_uri)

    def test_dest_local_copy(self, reset_db: None) -> None:
        """Test destination attempting to copy a local file to another destination"""

        # Create valid local file Paths and URIs for images
        dest_uri = self.resolve_destination_uri(StorageTarget.LOCAL_STORE)
        dest1_uri = f'{dest_uri}/bucket1'

        # The intent of this test is to copy the same image to two different destinations
        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        t.add_computed_column(img_rot1=t.img, destination=None)
        t.add_computed_column(img_rot2=t.img, destination=dest1_uri)
        print(t.collect())
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        r = t.collect()
        print(r)

        r_dest = t.select(t.img.fileurl, t.img_rot1.fileurl, t.img_rot2.fileurl).collect()
        print(r_dest)

        assert len(r) == 2

        if Env.get().default_output_media_dest is None:
            # Copying a local file to the LocalStore is not allowed
            assert ObjectOps.count(t._id, default_output_dest=True) == 0
        else:
            assert ObjectOps.count(t._id, default_output_dest=True) == len(r)

        # Ensure that local file is copied to a specified destination
        assert ObjectOps.count(t._id, dest=dest1_uri) == len(r)

    def test_dest_all(self, reset_db: None) -> None:
        """Test destination with all available storage targets"""
        dest_uris = tuple(self.resolve_destination_uri(dest_id) + '/bucket1' for dest_id in self.TESTED_DESTINATIONS)

        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        for i, (dest_id, dest_uri) in enumerate(zip(self.TESTED_DESTINATIONS, dest_uris, strict=True)):
            t.add_computed_column(**{f'img_rot_{dest_id}': t.img.rotate(30 * i)}, destination=dest_uri)
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])

        assert t.count() == 2

        # Query: verify original fileurls
        r_dest = t.select(
            t.img.fileurl, *[t[f'img_rot_{dest_id}'].fileurl for dest_id in self.TESTED_DESTINATIONS]
        ).collect()

        # Verify r_dest structure
        assert len(r_dest) == 2, f'Expected 2 rows, got {len(r_dest)}'

        # Validate that files belong to their destinations
        # Column 0 is input fileurl, columns 1+ are destination fileurls in TESTED_DESTINATIONS order
        dest_id_to_uri = dict(zip(self.TESTED_DESTINATIONS, dest_uris, strict=True))
        num_rows = len(r_dest)

        for row_idx in range(num_rows):
            # Column 0 is input fileurl (skip validation for input)
            # Columns 1+ correspond to destinations in TESTED_DESTINATIONS order
            for col_idx, dest_id in enumerate(self.TESTED_DESTINATIONS, start=1):
                expected_dest_uri = dest_id_to_uri[dest_id]
                # Access column by index: col_idx corresponds to dest_id
                col_name = list(r_dest.schema.keys())[col_idx]
                rotated_image_destination_url = r_dest[col_name][row_idx]

                # Parse the destination URL to verify it belongs to the expected destination
                destination_soa = ObjectPath.parse_object_storage_addr(
                    rotated_image_destination_url, allow_obj_name=True
                )

                # Verify the destination URL belongs to the expected destination
                assert destination_soa.storage_target == dest_id, (
                    f'Destination URL for {dest_id} should have storage_target {dest_id}, '
                    f'got {destination_soa.storage_target}'
                )

                # Check that the container/bucket matches the expected destination
                if destination_soa.container:
                    # Extract container from expected_dest_uri for comparison
                    expected_soa = ObjectPath.parse_object_storage_addr(expected_dest_uri, allow_obj_name=False)
                    assert destination_soa.container == expected_soa.container, (
                        f'Container {destination_soa.container} does not match expected container '
                        f'{expected_soa.container} for {dest_id}'
                    )

                # For local store, verify it's a file:// URL
                if dest_id == StorageTarget.LOCAL_STORE:
                    assert rotated_image_destination_url.startswith('file://'), (
                        f'Local store URL should start with file://, got {rotated_image_destination_url}'
                    )

        for uri in dest_uris:
            assert ObjectOps.count(t._id, dest=uri) == 2

        for uri in dest_uris:
            object_list = ObjectOps.list_uris(uri, n_max=20)
            assert len(object_list) >= 2

        pxt.drop_table(t)
        for uri in dest_uris:
            assert ObjectOps.count(t._id, dest=uri) == 0

    @rerun(reruns=3, reruns_delay=5)
    def test_presigned_url_all_destinations(self, reset_db: None) -> None:
        """Test presigned_url UDF for all cloud storage destinations"""
        # Exclude LOCAL_STORE as it doesn't support presigned URLs
        cloud_destinations = [d for d in self.TESTED_DESTINATIONS if d != StorageTarget.LOCAL_STORE]

        # Filter out destinations that aren't configured or fail to resolve
        available_destinations: list[StorageTarget] = []
        dest_uris: list[str] = []
        for dest_id in cloud_destinations:
            uri = self.resolve_destination_uri(dest_id, skip_on_failure=False)
            if uri is not None:
                available_destinations.append(dest_id)
                dest_uris.append(uri + '/bucket1')
        print('available_destinations: ', available_destinations)
        if not available_destinations:
            pytest.skip('No cloud destinations are configured or reachable')

        t = pxt.create_table('test_presigned_url', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        for i, (dest_id, dest_uri) in enumerate(zip(available_destinations, dest_uris, strict=True)):
            t.add_computed_column(**{f'img_rot_{dest_id}': t.img.rotate(30 * i)}, destination=dest_uri)
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])

        assert t.count() == 2

        # Query with presigned URLs
        expiration_seconds = 300
        r_dest_with_presigned = t.select(
            t.img.fileurl,
            *[presigned_url(t[f'img_rot_{dest_id}'].fileurl, expiration_seconds) for dest_id in available_destinations],
        ).collect()

        # Validate presigned URLs - same structure as first query in test_dest_all
        # Column 0 is input fileurl, columns 1+ are presigned URLs in available_destinations order
        num_rows = len(r_dest_with_presigned)

        # Track download failures for presigned URLs
        download_failures: list[tuple[StorageTarget, str]] = []

        for row_idx in range(num_rows):
            # Column 0 is input fileurl (skip validation for input)
            # Columns 1+ correspond to destinations in available_destinations order
            for col_idx, dest_id in enumerate(available_destinations, start=1):
                col_name = list(r_dest_with_presigned.schema.keys())[col_idx]
                presigned_url_str = r_dest_with_presigned[col_name][row_idx]

                # Presigned URLs should be HTTP/HTTPS
                assert presigned_url_str.startswith('http://') or presigned_url_str.startswith('https://'), (
                    f'Presigned URL for {dest_id} should be HTTP/HTTPS, got {presigned_url_str}'
                )

                # Download and verify the presigned URL
                success, error_msg = self._download_presigned_urls(presigned_url_str, dest_id)
                if not success:
                    download_failures.append((dest_id, error_msg))

        # Fail test at the end if any downloads failed
        if download_failures:
            failure_summary = '\n'.join([f'{dest_id}: {error_msg}' for dest_id, error_msg in download_failures])
            pytest.fail(
                f'Failed to download presigned URLs for {len(download_failures)} destination(s):\n{failure_summary}'
            )

        pxt.drop_table(t)

    def _download_presigned_urls(self, presigned_url_str: str, dest_id: StorageTarget) -> tuple[bool, str]:
        """Download and verify a presigned URL using requests.get

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            response = requests.get(presigned_url_str, timeout=10)
            if response.status_code != 200:
                error_msg = f'HTTP {response.status_code}'
                if response.text:
                    error_msg += f'\nResponse: {response.text}'
                return False, error_msg
            downloaded_data = response.content
            if len(downloaded_data) == 0:
                return False, 'Downloaded file is empty'
            return True, ''
        except requests.exceptions.RequestException as e:
            return False, f'Request exception: {e}'

    def __download_object(self, src_base: str, src_obj: str) -> None:
        """Test downloading a media object from a public Store"""
        src_uri = src_base + src_obj
        # Download a media object from Azure Blob Storage
        temp_path = TempStore.create_path()
        ObjectOps.copy_object_to_local_file(src_uri, temp_path)

        # Check that the file was downloaded successfully
        assert temp_path.exists()
        assert temp_path.stat().st_size > 0
        print(f'\nDownloaded: {temp_path}, {temp_path.stat().st_size}')

        # Clean up the temporary file
        temp_path.unlink()

        r = ObjectOps.list_objects(src_base, return_uri=True, n_max=20)
        print(f'List of objects in {src_base}:')
        for item in r:
            print(item)
        assert len(r) > 2

    PUBLIC_TEST_OBJECTS: ClassVar[dict[StorageTarget, tuple[str, str, str]]] = {
        # StorageTarget -> (module_name, src_base, src_obj)
        StorageTarget.AZURE_STORE: (
            'azure.storage.blob',
            'https://azureopendatastorage.blob.core.windows.net/mnist/',
            'train-images-idx3-ubyte.gz',
        ),
        StorageTarget.GCS_STORE: (
            'google.cloud.storage',
            'gs://hdrplusdata/',
            '20171106_subset/gallery_20171023/c483_20150901_105412_265.jpg',
        ),
        StorageTarget.S3_STORE: ('boto3', 's3://open-images-dataset/validation/', '3c02ca9ec9b2b77b.jpg'),
    }

    @pytest.mark.parametrize('dest_id', PUBLIC_TEST_OBJECTS.keys())
    def test_public_download(self, reset_db: None, dest_id: StorageTarget) -> None:
        """Test downloading a media object from a public Store"""
        module_name, src_base, src_obj = self.PUBLIC_TEST_OBJECTS[dest_id]
        skip_test_if_not_installed(module_name)
        self.__download_object(src_base, src_obj)
