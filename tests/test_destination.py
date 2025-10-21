from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional
from uuid import UUID

import pytest

import pixeltable as pxt
from pixeltable.config import Config
from pixeltable.utils.object_stores import ObjectOps, ObjectPath


class TestDestination:
    @staticmethod
    def validate_dest(dest: Optional[str]) -> bool:
        ObjectOps.validate_destination(dest, '')
        return True

    USE_LOCAL_DEST = 'file'
    USE_GS_DEST = 'gs'
    USE_S3_DEST = 's3'
    USE_R2_DEST = 'r2'
    USE_B2_DEST = 'b2'
    USE_AZURE_DEST = 'az'

    @staticmethod
    def create_destination_uri(n: int, dest_id: str) -> str:
        """
        Return the destination directory for test images
        """
        if dest_id == 'file':
            base_path = Config.get().home / 'test_dest'
            base_path.mkdir(exist_ok=True)
            dest_path = base_path / f'img_rot{n}'
            dest_path.mkdir(exist_ok=True)
            return dest_path.resolve().as_uri()
        if dest_id == 'gs':
            return f'gs://pxt-gs-test/pytest/img_rot{n}'
        elif dest_id == 's3':
            return f's3://pxt-test/pytest/img_rot{n}'
        elif dest_id == 'r2':
            return f'https://ae60fad96d33636287c3b2e76b88241f.r2.cloudflarestorage.com/pxt-test/pytest/img_rot{n}'
        elif dest_id == 'r2_bad':
            return f'https://a711169187abcf395c01dca4390ee0ea.r2.cloudflarestorage.com/pxt-test/pytest/img_rot{n}'
        elif dest_id == 'b2':
            return f'https://s3.us-east-005.backblazeb2.com/5f1d8127cd78c3509c9e0f17/pytest/img_rot{n}'
        raise AssertionError(f'Invalid dest_id: {dest_id}')

    @classmethod
    def get_valid_dest(cls, n: int, dest_id: str, backup_dest: str) -> str:
        """If the specified destination is not valid (no credentials), use the backup destination"""
        try:
            dest = cls.create_destination_uri(n, dest_id)
            ObjectOps.validate_destination(dest, '')
            return dest
        except Exception:
            return backup_dest

    @classmethod
    def count(cls, uri: Optional[str], tbl_id: UUID, tbl_version: Optional[int] = None) -> int:
        """Return the count of media files in the destination for a given table ID"""
        return ObjectOps.count(uri, tbl_id, tbl_version)

    def pr_us(self, us: pxt.UpdateStatus, op: str = '') -> None:
        """Print contents of UpdateStatus"""
        print(f'=========================== pr_us =========================== op: {op}')
        print(f'num_rows: {us.num_rows}')
        print(f'num_computed_values: {us.num_computed_values}')
        print(f'num_excs: {us.num_excs}')
        print(f'updated_cols: {us.updated_cols}')
        print(f'cols_with_excs: {us.cols_with_excs}')
        print(us.row_count_stats)
        print(us.cascade_row_count_stats)
        print('============================================================')

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
        r2_bad_dest = self.create_destination_uri(1, 'r2_bad')
        assert not self.validate_dest(r2_bad_dest)

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
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000030.jpg', allow_obj_name=True
        )

        ObjectPath.parse_object_storage_addr('file://dir1/dir2/dir3', allow_obj_name=False)
        ObjectPath.parse_object_storage_addr(f'file://dir1/dir2/dir3/{o_name}', allow_obj_name=True)
        ObjectPath.parse_object_storage_addr(f'dir2/dir3/{o_name}', allow_obj_name=True)

    @pytest.mark.parametrize('dest_id', ['file', 'gs', 's3', 'r2', 'b2'])
    def test_destination(self, reset_db: None, dest_id: str) -> None:
        """Test various media destinations."""
        if not self.validate_dest(self.create_destination_uri(1, dest_id)):
            pytest.skip(f'Destination {dest_id} not installed or not reachable')

        dest1_uri = self.create_destination_uri(1, dest_id)
        dest2_uri = self.create_destination_uri(2, dest_id)

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
        assert n == self.count(None, t._id)
        assert n == self.count(dest1_uri, t._id)
        assert n == self.count(dest2_uri, t._id)

        n = 1
        assert n == self.count(None, t._id, 2)
        assert n == self.count(dest1_uri, t._id, 3)
        assert n == self.count(dest2_uri, t._id, 4)

        version = 5
        n = 1
        assert n == self.count(None, t._id, version)
        assert n == self.count(dest1_uri, t._id, version)
        assert n == self.count(dest2_uri, t._id, version)

        # Test that we can list objects in the destination
        olist = ObjectOps.list_uris(dest1_uri, n_max=10)
        print('list of files in the destination')
        for item in olist:
            print(item)
        assert len(olist) >= 2

        # Ensure that all media is removed when the table is dropped
        save_id = t._id
        pxt.drop_table(t)

        assert self.count(None, save_id) == 0
        assert self.count(dest1_uri, save_id) == 0
        assert self.count(dest2_uri, save_id) == 0

    @pytest.mark.parametrize('dest_id', ['file', 'gs', 's3', 'r2', 'b2'])
    def test_dest_two_copies(self, reset_db: None, dest_id: str) -> None:
        """Test destination with two Stores receiving copies of the same computed image"""
        if not self.validate_dest(self.create_destination_uri(1, dest_id)):
            pytest.skip(f'Destination {dest_id} not installed or not reachable')

        # Create two valid local file Paths for images
        dest1_uri = self.create_destination_uri(1, dest_id)
        dest2_uri = self.create_destination_uri(2, dest_id)

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
        assert len(r) == self.count(None, t._id)
        assert len(r) == self.count(dest1_uri, t._id)

        # The outcome of this test is unusual:
        # When the column img_rot4 is ADDED, the computed result for existing rows is not identified
        # as a duplicate, so it is double copied to the destination.
        # When new rows are INSERTED, the results and destinations for img_rot3 and img_rot4 are identified
        # as duplicates, so they are not double copied to the destination.
        assert len(r) + 1 == self.count(dest2_uri, t._id)

    def test_dest_local_copy(self, reset_db: None) -> None:
        """Test destination attempting to copy a local file to another destination"""

        # Create valid local file Paths and URIs for images
        dest1_uri = self.create_destination_uri(1, 'file')

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
        assert self.count(None, t._id) == 0

        # Ensure that local file is copied to a specified destination
        assert len(r) == self.count(dest1_uri, t._id)

    def test_dest_all(self, reset_db: None) -> None:
        """Test destination with all available storage targets"""
        n = 1
        lc_uri = self.create_destination_uri(n, self.USE_LOCAL_DEST)
        c2_uri = self.get_valid_dest(n, self.USE_GS_DEST, lc_uri)
        c3_uri = self.get_valid_dest(n, self.USE_S3_DEST, lc_uri)
        c4_uri = self.get_valid_dest(n, self.USE_R2_DEST, lc_uri)
        c5_uri = self.get_valid_dest(n, self.USE_B2_DEST, lc_uri)
        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        t.add_computed_column(img_rot_1=t.img.rotate(90), destination=lc_uri)
        t.add_computed_column(img_rot_2=t.img.rotate(180), destination=c2_uri)
        t.add_computed_column(img_rot_3=t.img.rotate(270), destination=c3_uri)
        t.add_computed_column(img_rot_4=t.img.rotate(360), destination=c4_uri)
        t.add_computed_column(img_rot_5=t.img.rotate(450), destination=c5_uri)
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])

        tbl_id = t._id
        assert t.count() == 2
        target_count: dict[str, int] = defaultdict(int)
        print(f'Using destinations:\n  {lc_uri}\n  {c2_uri}\n  {c3_uri}\n  {c4_uri}\n  {c5_uri}')
        r_dest = t.select(
            t.img.fileurl,
            t.img_rot_1.fileurl,
            t.img_rot_2.fileurl,
            t.img_rot_3.fileurl,
            t.img_rot_4.fileurl,
            t.img_rot_5.fileurl,
        ).collect()
        print(r_dest)
        for t_uri in [lc_uri, c2_uri, c3_uri, c4_uri, c5_uri]:
            print(f'Count for {t_uri}: {self.count(t_uri, tbl_id)}')
            target_count[t_uri] += 2
        for t_uri in [lc_uri, c2_uri, c3_uri, c4_uri, c5_uri]:
            assert self.count(t_uri, tbl_id) == target_count[t_uri], f'Count mismatch for {t_uri}'

        for t_uri in [lc_uri, c2_uri, c3_uri, c4_uri, c5_uri]:
            olist = ObjectOps.list_uris(t_uri, n_max=20)
            print('list of files in the destination')
            for item in olist:
                print(item)
            assert len(olist) >= 2

        pxt.drop_table(t)
        for t_uri in [lc_uri, c2_uri, c3_uri, c4_uri, c5_uri]:
            assert self.count(t_uri, tbl_id) == 0

    def dest_public_read_only(self, src_base: str, src_obj: str) -> None:
        """Test downloading a media object from a public Store"""
        from pixeltable.utils.local_store import TempStore

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

    S3_PUBLIC_BUCKET = 's3://open-images-dataset/validation/'
    S3_PUBLIC_OBJECT = '3c02ca9ec9b2b77b.jpg'

    def test_dest_public_s3(self, reset_db: None) -> None:
        """Test s3 interfaces on public bucket / object"""
        if not self.validate_dest(self.S3_PUBLIC_BUCKET):
            pytest.skip('S3 support not installed or destination not reachable')
        self.dest_public_read_only(self.S3_PUBLIC_BUCKET, self.S3_PUBLIC_OBJECT)

    GS_PUBLIC_BUCKET = 'gs://hdrplusdata/'
    GS_PUBLIC_OBJECT = '20171106_subset/gallery_20171023/c483_20150901_105412_265.jpg'

    def test_dest_public_gs(self, reset_db: None) -> None:
        """Test Google Cloud Storage interfaces on public bucket / object"""
        if not self.validate_dest(self.GS_PUBLIC_BUCKET):
            pytest.skip('GS support not installed or destination not reachable')
        self.dest_public_read_only(self.GS_PUBLIC_BUCKET, self.GS_PUBLIC_OBJECT)

    AZ_PUBLIC_BUCKET = 'https://azureopendatastorage.blob.core.windows.net/mnist/'
    AZ_PUBLIC_OBJECT = 'train-images-idx3-ubyte.gz'

    def test_dest_public_az(self, reset_db: None) -> None:
        """Test Azure interfaces on public bucket / object"""
        if not self.validate_dest(self.AZ_PUBLIC_BUCKET):
            pytest.skip('AZ support not installed or destination not reachable')
        self.dest_public_read_only(self.AZ_PUBLIC_BUCKET, self.AZ_PUBLIC_OBJECT)
