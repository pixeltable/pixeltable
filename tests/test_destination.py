from pathlib import Path

import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import env
from pixeltable.utils.media_store import MediaStore


class TestDestination:
    @classmethod
    def base_dest(cls) -> Path:
        """Return the base destination directory for tests"""
        base_path = env.Env.get().tmp_dir / '..' / 'test_dest'
        base_path.mkdir(exist_ok=True)
        return base_path

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

        # Create two valid local file Paths for image rotation
        valid_dest_1 = self.base_dest() / 'img_rot1'
        if not valid_dest_1.exists():
            valid_dest_1.mkdir()
        valid_dest_2 = self.base_dest() / 'img_rot2'
        if not valid_dest_2.exists():
            valid_dest_2.mkdir()

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

        dest1_uri = valid_dest_1.resolve().as_uri()
        dest2_uri = valid_dest_2.resolve().as_uri()
        assert len(r) == 2
        assert len(r) == MediaStore.get().count(t._id)
        assert len(r) == MediaStore.get(dest1_uri).count(t._id)
        assert len(r) == MediaStore.get(dest2_uri).count(t._id)

    def test_dest_local_3x3(self, reset_db: None) -> None:
        """Test destination with two local Paths receiving copies of the same computed image"""

        # Create two valid local Paths for image rotation
        valid_dest_1 = self.base_dest() / 'img_rot1'
        if not valid_dest_1.exists():
            valid_dest_1.mkdir()
        valid_dest_2 = self.base_dest() / 'img_rot2'
        if not valid_dest_2.exists():
            valid_dest_2.mkdir()

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

        dest1_uri = valid_dest_1.resolve().as_uri()
        dest2_uri = valid_dest_2.resolve().as_uri()
        assert len(r) == 2
        assert len(r) == MediaStore.get().count(t._id)
        assert len(r) == MediaStore.get(dest1_uri).count(t._id)

        # The outcome of this test is unusual:
        # When the column img_rot4 is ADDED, the computed result for existing rows is not identified
        # as a duplicate, so it is double copied to the destination.
        # When new rows are INSERTED, the results and destinations for img_rot3 and img_rot4 are identified
        # as duplicates, so they are not double copied to the destination.
        assert len(r) + 1 == MediaStore.get(dest2_uri).count(t._id)

    def test_dest_local_uri(self, reset_db: None) -> None:
        """Test destination with local URI"""

        # Create a valid local directories for image rotation, expressed as file URI
        valid_dest_1 = self.base_dest() / 'img_rot1'
        if not valid_dest_1.exists():
            valid_dest_1.mkdir()
        dest1_uri = valid_dest_1.resolve().as_uri()

        t = pxt.create_table('test_dest', schema={'img': pxt.Image})
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        t.add_computed_column(img_rot1=t.img.rotate(90), destination=None)
        t.add_computed_column(img_rot2=t.img.rotate(180), destination=valid_dest_1)
        print(t.collect())
        t.insert([{'img': 'tests/data/imagenette2-160/ILSVRC2012_val_00000557.JPEG'}])
        r = t.collect()
        print(r)

        assert len(r) == 2
        assert len(r) == MediaStore.get().count(t._id)
        assert len(r) == MediaStore.get(dest1_uri).count(t._id)

    def test_dest_local_copy(self, reset_db: None) -> None:
        """Test destination attempting to copy a local file to another destination"""

        # Create two valid local file Paths for image rotation
        valid_dest_1 = self.base_dest() / 'img_rot1'
        if not valid_dest_1.exists():
            valid_dest_1.mkdir()

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

        dest1_uri = valid_dest_1.resolve().as_uri()

        # Copying a local file to the MediaStore is not allowed
        assert MediaStore.get().count(t._id) == 0

        # Ensure that local file is copied to a specified destination
        assert len(r) == MediaStore.get(dest1_uri).count(t._id)
