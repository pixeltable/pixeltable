import pytest

from pixeltable.utils.media_store import MediaStore
import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import env


class TestDestination:
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

        # Create two valid local directories for image rotation
        base_dest = env.Env.get().tmp_dir
        valid_dest_1 = base_dest / 'img_rot1'
        if not valid_dest_1.exists():
            valid_dest_1.mkdir()
        valid_dest_2 = base_dest / 'img_rot2'
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

        assert len(r) == 2
        assert 3 * len(r) == MediaStore.get().count(t._id)
#        assert len(r) == MediaStore.get(valid_dest_1).count(t._id)
#        assert len(r) == MediaStore.get(valid_dest_2).count(t._id)
