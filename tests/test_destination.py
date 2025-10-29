from __future__ import annotations

import os
from typing import ClassVar

import pytest

import pixeltable as pxt
from pixeltable.config import Config
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import ObjectOps, ObjectPath, StorageTarget

from .utils import skip_test_if_not_installed


class TestDestination:
    TESTED_DESTINATIONS = (
        StorageTarget.AZURE_STORE,
        StorageTarget.B2_STORE,
        StorageTarget.GCS_STORE,
        StorageTarget.LOCAL_STORE,
        StorageTarget.R2_STORE,
        StorageTarget.S3_STORE,
    )

    @classmethod
    def resolve_destination_uri(cls, dest_id: StorageTarget) -> str:
        assert dest_id in cls.TESTED_DESTINATIONS
        uri: str
        match dest_id:
            case StorageTarget.AZURE_STORE:
                uri = 'https://pixeltable1.blob.core.windows.net/pytest'
            case StorageTarget.B2_STORE:
                uri = 'https://s3.us-east-005.backblazeb2.com/pixeltable/pytest'
            case StorageTarget.GCS_STORE:
                if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
                    pytest.skip('GOOGLE_APPLICATION_CREDENTIALS is not set')
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

        try:
            ObjectOps.validate_destination(uri)
            return uri
        except Exception as exc:
            pytest.skip(f'Destination {str(dest_id)!r} not reachable or not configured properly: {exc}')

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

        # Test with a destination that is not reachable
        with pytest.raises(Exception):  # noqa: B017
            ObjectOps.validate_destination(
                'https://a711169187abcf395c01dca4390ee0ea.r2.cloudflarestorage.com/pxt-test/pytest'
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

        n = len(r)
        assert n == 2
        assert n == ObjectOps.count(None, t._id)
        assert n == ObjectOps.count(dest1_uri, t._id)
        assert n == ObjectOps.count(dest2_uri, t._id)

        n = 1
        assert n == ObjectOps.count(None, t._id, 2)
        assert n == ObjectOps.count(dest1_uri, t._id, 3)
        assert n == ObjectOps.count(dest2_uri, t._id, 4)

        version = 5
        n = 1
        assert n == ObjectOps.count(None, t._id, version)
        assert n == ObjectOps.count(dest1_uri, t._id, version)
        assert n == ObjectOps.count(dest2_uri, t._id, version)

        # Test that we can list objects in the destination
        olist = ObjectOps.list_uris(dest1_uri, n_max=10)
        print('list of files in the destination')
        for item in olist:
            print(item)
        assert len(olist) >= 2

        # Ensure that all media is removed when the table is dropped
        save_id = t._id
        pxt.drop_table(t)

        assert ObjectOps.count(None, save_id) == 0
        assert ObjectOps.count(dest1_uri, save_id) == 0
        assert ObjectOps.count(dest2_uri, save_id) == 0

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
        assert len(r) == ObjectOps.count(None, t._id)
        assert len(r) == ObjectOps.count(dest1_uri, t._id)

        # The outcome of this test is unusual:
        # When the column img_rot4 is ADDED, the computed result for existing rows is not identified
        # as a duplicate, so it is double copied to the destination.
        # When new rows are INSERTED, the results and destinations for img_rot3 and img_rot4 are identified
        # as duplicates, so they are not double copied to the destination.
        assert len(r) + 1 == ObjectOps.count(dest2_uri, t._id)

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

        # Copying a local file to the LocalStore is not allowed
        assert ObjectOps.count(None, t._id) == 0

        # Ensure that local file is copied to a specified destination
        assert len(r) == ObjectOps.count(dest1_uri, t._id)

    def test_dest_all(self, reset_db: None) -> None:
        """Test destination with all available storage targets"""
        dest_uris = tuple(self.resolve_destination_uri(dest_id) + '/bucket1' for dest_id in self.TESTED_DESTINATIONS)

        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        for i, (dest_id, dest_uri) in enumerate(zip(self.TESTED_DESTINATIONS, dest_uris, strict=True)):
            t.add_computed_column(**{f'img_rot_{dest_id}': t.img.rotate(30 * i)}, destination=dest_uri)
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])

        assert t.count() == 2
        r_dest = t.select(
            t.img.fileurl, *[getattr(t, f'img_rot_{dest_id}').fileurl for dest_id in self.TESTED_DESTINATIONS]
        ).collect()
        print(r_dest)
        for uri in dest_uris:
            print(f'Count for {uri}: {ObjectOps.count(uri, t._id)}')
            assert ObjectOps.count(uri, t._id) == 2

        for uri in dest_uris:
            object_list = ObjectOps.list_uris(uri, n_max=20)
            assert len(object_list) >= 2

        pxt.drop_table(t)
        for uri in dest_uris:
            assert ObjectOps.count(uri, t._id) == 0

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
