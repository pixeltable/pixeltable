from pathlib import Path
from typing import Optional, Union
from uuid import UUID

import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import env
from pixeltable.utils.media_destination import MediaDestination
from pixeltable.utils.media_path import MediaPath


class TestDestination:
    USE_S3 = False
    USE_GS = False
    USE_R2 = False

    @classmethod
    def base_dest(cls) -> Path:
        """Return the base destination directory for tests"""
        base_path = env.Env.get().tmp_dir / '..' / 'test_dest'
        base_path.mkdir(exist_ok=True)
        return base_path

    @classmethod
    def dest(cls, n: int) -> tuple[Union[Path, str], str]:
        """Return the destination directory for test images"""
        if cls.USE_GS:
            gs_uri = f'gs://pixeltable/my_folder/img_rot{n}'
            return gs_uri, gs_uri
        elif cls.USE_S3:
            s3_uri = f's3://jimpeterson-test/img_rot{n}'
            return s3_uri, s3_uri
        elif cls.USE_R2:
            r2_uri = f'https://a711169187ea0f395c01dca4390ee0ea.r2.cloudflarestorage.com/jimpeterson-testr2/images/img_rot{n}'
            return r2_uri, r2_uri
        else:
            dest_path = cls.base_dest() / f'img_rot{n}'
            if not dest_path.exists():
                dest_path.mkdir()
            dest_uri = dest_path.resolve().as_uri()
            return dest_path, dest_uri

    @classmethod
    def count(cls, uri: Optional[str], tbl_id: UUID, tbl_version: Optional[int] = None) -> int:
        """Return the count of media files in the destination for a given table ID"""
        return MediaDestination.count(uri, tbl_id, tbl_version)

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
        with pytest.raises(excs.Error, match='must be a string or path'):
            t.add_computed_column(img_rot=t.img.rotate(90), destination=27)

        with pytest.raises(excs.Error, match='only applies to stored computed columns'):
            t.add_computed_column(img_rot=t.img.rotate(90), stored=False, destination=valid_dest)

        with pytest.raises(excs.Error, match='only applies to stored computed columns'):
            _ = pxt.create_table('test_dest_bad', schema={'img': {'type': pxt.Image, 'destination': f'{valid_dest}'}})

        # Test destination with a non-existent directory
        with pytest.raises(excs.Error, match='does not exist'):
            t.add_computed_column(img_rot=t.img.rotate(90), destination='non_existent_dir/img_rot')

        # Test destination with a file path instead of a directory
        with pytest.raises(excs.Error, match='must be a directory, not a file'):
            t.add_computed_column(
                img_rot=t.img.rotate(90), destination='tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'
            )

        # Test with invalid scheme
        with pytest.raises(excs.Error, match='must be a valid URI to a supported'):
            t.add_computed_column(img_rot=t.img.rotate(90), destination='https://anything/')

    def parse_one(self, s: str, consider_object: bool) -> bool:
        print(f'Parsing: {s}')
        try:
            r = MediaPath.parse_media_storage_addr(s, consider_object)
            print(f'Success: {r!r}')
            return True
        except ValueError as e:
            print(f'Error: {e!r}')
            return False

    def parse_two(self, s: str) -> bool:
        r = self.parse_one(s, False)
        r &= self.parse_one(s, True)
        return r

    def test_dest_parser(self, reset_db: None) -> None:
        a_name = 'acct-name'
        o_name = 'obj-name'
        p_name1 = 'path-name'
        p_name2 = 'path-name/path2-name'
        r = True
        for s in [
            's3://container',
            f'wasb://container@{a_name}.blob.core.windows.net',
            f'https://{a_name}.blob.core.windows.net/container',
            f'https://{a_name}.r2.cloudflarestorage.com/container',
            'https://raw.github.com',
            'file://dir1/dir2/dir3',
            'dir1/dir2/dir3',
        ]:
            r &= self.parse_two(s)
            r &= self.parse_two(s + '/')
            r &= self.parse_two(s + '/' + p_name1)
            r &= self.parse_two(s + '/' + p_name1 + '/')
            r &= self.parse_two(s + '/' + p_name2)
            r &= self.parse_two(s + '/' + p_name2 + '/')
            r &= self.parse_two(s + '/' + o_name)
            r &= self.parse_two(s + '/' + p_name2 + '/' + o_name)

        r &= self.parse_one(
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000030.jpg', True
        )

        r &= self.parse_one('file://dir1/dir2/dir3', False)
        r &= self.parse_one(f'file://dir1/dir2/dir3/{o_name}', True)
        r &= self.parse_one(f'dir2/dir3/{o_name}', True)

        assert r

    #        assert False

    def test_dest_local_2(self, reset_db: None) -> None:
        """Test destination with two local directories"""

        # Create two valid local file Paths for images
        valid_dest_1, dest1_uri = self.dest(1)
        valid_dest_2, dest2_uri = self.dest(2)

        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        t.add_computed_column(img_rot1=t.img.rotate(90), destination=None)
        t.add_computed_column(img_rot2=t.img.rotate(180), destination=valid_dest_1)
        t.add_computed_column(img_rot3=t.img.rotate(270), destination=valid_dest_2)
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

        # Ensure that all media is removed when the table is dropped
        save_id = t._id
        pxt.drop_table(t)

        assert self.count(None, save_id) == 0
        assert self.count(dest1_uri, save_id) == 0
        assert self.count(dest2_uri, save_id) == 0

    #        assert None == 'We made it to the end'

    def test_dest_local_3x3(self, reset_db: None) -> None:
        """Test destination with two local Paths receiving copies of the same computed image"""

        # Create two valid local file Paths for images
        valid_dest_1, dest1_uri = self.dest(1)
        valid_dest_2, dest2_uri = self.dest(2)

        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        t.add_computed_column(img_rot1=t.img.rotate(90), destination=None)
        t.add_computed_column(img_rot2=t.img.rotate(90), destination=valid_dest_1)
        t.add_computed_column(img_rot3=t.img.rotate(90), destination=valid_dest_2)
        t.add_computed_column(img_rot4=t.img.rotate(90), destination=valid_dest_2)  # Try to copy twice to the same dest
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

    def test_dest_local_uri(self, reset_db: None) -> None:
        """Test destination with local URI"""

        # Create valid local file Paths and URIs for images
        _, dest1_uri = self.dest(1)

        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        t.add_computed_column(img_rot1=t.img.rotate(90), destination=None)
        t.add_computed_column(img_rot2=t.img.rotate(180), destination=dest1_uri)
        print(t.collect())
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        r = t.collect()
        print(r)

        assert len(r) == 2
        assert len(r) == self.count(None, t._id)
        assert len(r) == self.count(dest1_uri, t._id)

    def test_dest_local_copy(self, reset_db: None) -> None:
        """Test destination attempting to copy a local file to another destination"""

        # Create valid local file Paths and URIs for images
        valid_dest_1, dest1_uri = self.dest(1)

        # The intent of this test is to copy the same image to two different destinations
        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        t.add_computed_column(img_rot1=t.img, destination=None)
        t.add_computed_column(img_rot2=t.img, destination=valid_dest_1)
        print(t.collect())
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        r = t.collect()
        print(r)

        r_dest = t.select(t.img.fileurl, t.img_rot1.fileurl, t.img_rot2.fileurl).collect()
        print(r_dest)

        assert len(r) == 2

        # Copying a local file to the MediaStore is not allowed
        assert self.count(None, t._id) == 0

        # Ensure that local file is copied to a specified destination
        assert len(r) == self.count(dest1_uri, t._id)

    def test_dest_write_perf(self, reset_db: None) -> None:
        """Test write performance with multiple concurrent requests"""

        img_data = 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'  # ~10 KB
        # img_data = 'tests/data/images/sewing-threads.heic'  # ~1.5 MB
        # img_data = 'tests/data/imagenette2-160/ILSVRC2012_val_00015787.JPEG'  # ~3 KB
        data_rows = 10
        number_of_inserts = 1

        s3_cols = 0
        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        for i in range(1, 4):
            _, dest_uri = self.dest(i)
            t.add_computed_column(**{f'img_rot_1_{i}': t.img.rotate(90)}, destination=dest_uri)
            t.add_computed_column(**{f'img_rot_2_{i}': t.img.rotate(180)}, destination=dest_uri)
            t.add_computed_column(**{f'img_rot_3_{i}': t.img.rotate(270)}, destination=dest_uri)
            s3_cols += 3
        data = [{'img': img_data} for _ in range(data_rows)]
        for _ in range(number_of_inserts):
            t.insert(data)
        n = t.count()

        assert n == number_of_inserts * data_rows
        for i in range(1, 4):
            _, dest_uri = self.dest(i)
            assert 3 * n == self.count(dest_uri, t._id)

        # Ensure that all media is removed when the table is dropped
        save_id = t._id
        pxt.drop_table(t)
        for i in range(1, 4):
            _, dest_uri = self.dest(i)
            assert self.count(dest_uri, save_id) == 0

    def test_dest_list(self, reset_db: None) -> None:
        """Test destination listing with GCS URIs"""

        # Create valid GCS URIs for images
        _, dest1_uri = self.dest(1)

        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        t.add_computed_column(img_rot1=t.img.rotate(90), destination=None)
        t.add_computed_column(img_rot2=t.img.rotate(180), destination=dest1_uri)

        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        r = t.collect()
        print(r)

        r_dest = t.select(t.img.fileurl, t.img_rot1.fileurl, t.img_rot2.fileurl).collect()
        print(r_dest)

        assert len(r) == 2
        assert len(r) == self.count(None, t._id)
        assert len(r) == self.count(dest1_uri, t._id)

        olist = MediaDestination.list_uris(dest1_uri, n_max=10)
        print('list of files in the destination')
        for item in olist:
            print(item)
        assert len(olist) >= 2
