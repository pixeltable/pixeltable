import PIL.Image

import pixeltable as pxt

from ..utils import SAMPLE_IMAGE_URL


class TestImage:
    def test_tile_iterator(self, reset_db):
        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.insert(image=SAMPLE_IMAGE_URL)
        v = pxt.create_view(
            'test_view',
            t,
            iterator=pxt.iterators.TileIterator.create(image=t.image, tile_size=(100, 100), overlap=(10, 10))
        )
        image: PIL.Image.Image = t.collect()[0]['image']
        results = v.select(v.tile, v.tile_coord, v.tile_box).order_by(v.pos).collect()
        assert image.size == (640, 480)
        assert len(results) == 42
        for j in range(6):
            for i in range(7):
                result = results[j * 7 + i]
                assert result['tile_coord'] == [i, j]
                box = [i * 90, j * 90, 100 + i * 90, 100 + j * 90]
                assert result['tile_box'] == box
                assert result['tile'].size == (100, 100)
                tile = image.crop(box)
                assert list(result['tile'].getdata()) == list(tile.getdata())
