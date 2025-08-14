from pathlib import Path
from typing import Optional, Union
from uuid import UUID

import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import env
from pixeltable.utils.media_store import MediaDestination


class TestDestination:
    USE_S3 = False

    @classmethod
    def base_dest(cls) -> Path:
        """Return the base destination directory for tests"""
        base_path = env.Env.get().tmp_dir / '..' / 'test_dest'
        base_path.mkdir(exist_ok=True)
        return base_path

    @classmethod
    def dest(cls, n: int) -> tuple[Union[Path, str], str]:
        """Return the destination directory for test images"""
        if cls.USE_S3:
            s3_uri = f's3://jimpeterson-test/img_rot{n}'
            return s3_uri, s3_uri
        else:
            dest_path = cls.base_dest() / f'img_rot{n}'
            if not dest_path.exists():
                dest_path.mkdir()
            dest_uri = dest_path.resolve().as_uri()
            return dest_path, dest_uri

    @classmethod
    def count(cls, uri: Optional[str], tbl_id: UUID) -> int:
        """Return the count of media files in the destination for a given table ID"""
        return MediaDestination.count(uri, tbl_id)

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

        assert len(r) == 2
        assert len(r) == self.count(None, t._id)
        assert len(r) == self.count(dest1_uri, t._id)
        assert len(r) == self.count(dest2_uri, t._id)

        # Ensure that all media is removed when the table is dropped
        save_id = t._id
        pxt.drop_table(t)

        assert self.count(None, save_id) == 0
        assert self.count(dest1_uri, save_id) == 0
        assert self.count(dest2_uri, save_id) == 0

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

        s3_cols = 0
        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        for i in range(1, 4):
            _, dest_uri = self.dest(i)
            t.add_computed_column(**{f'img_rot_1_{i}': t.img.rotate(90)}, destination=dest_uri)
            t.add_computed_column(**{f'img_rot_2_{i}': t.img.rotate(180)}, destination=dest_uri)
            t.add_computed_column(**{f'img_rot_3_{i}': t.img.rotate(270)}, destination=dest_uri)
            s3_cols += 3
        data_rows = 1000
        data = [{'img': img_data} for _ in range(data_rows)]
        inserts = 1
        for _ in range(inserts):
            t.insert(data)
        n = t.count()

        assert n == inserts * data_rows
        for i in range(1, 4):
            _, dest_uri = self.dest(i)
            assert 3 * n == self.count(dest_uri, t._id)

        # Ensure that all media is removed when the table is dropped
        save_id = t._id
        pxt.drop_table(t)
        for i in range(1, 4):
            _, dest_uri = self.dest(i)
            assert self.count(dest_uri, save_id) == 0
