import re

import PIL.Image
import pytest
from PIL.Image import Dither, Image, Quantize, Transpose

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.functions.image import alpha_composite, blend, composite, stitch_tiles, tile_iterator

from ..utils import SAMPLE_IMAGE_URL, get_image_files, pxt_raises

pytestmark = pytest.mark.local('UDF/integration test')


class TestImage:
    def test_image(self, img_tbl: pxt.Table) -> None:
        # mask_img = next(f for f in get_image_files() if f.endswith('n03888257_1389.JPEG'))
        t = img_tbl
        _ = t.select(t.img.rotate(90)).show()
        # alpha_composite needs RGBA images to work, so we do the conversions inline
        _ = t.select(alpha_composite(t.img.convert(mode='RGBA'), t.img.rotate(90).convert(mode='RGBA'))).show()
        _ = t.select(blend(t.img, t.img.rotate(90), 0.5)).show()
        _ = t.select(composite(t.img, t.img.rotate(90), mask=t.img.convert(mode='RGBA'))).show()
        _ = t.select(t.img.crop((0, 0, 10, 10))).show()
        _ = t.select(t.img.getchannel(0)).show()
        _ = t.select(t.img.resize([100, 100])).show()
        _ = t.select(t.img.effect_spread(3)).show()
        _ = t.select(t.img.entropy(mask=t.img.convert(mode='L'))).show()
        _ = t.select(t.img.getbands()).show()
        _ = t.select(t.img.getbbox()).show()
        _ = t.select(t.img.getcolors(5)).show()
        _ = t.select(t.img.getextrema()).show()
        _ = t.select(t.img.getpalette()).show()
        _ = t.select(t.img.getpixel([5, 5])).show()
        _ = t.select(t.img.getprojection()).show()
        _ = t.select(t.img.histogram(mask=t.img.convert(mode='L'))).show()
        _ = t.select(t.img.convert(mode='1').point([min(255, x * 20) for x in range(256)])).show()
        _ = t.select(t.img.quantize()).show()
        _ = t.select(t.img.quantize(256, Quantize.MEDIANCUT, 3, None, Dither.NONE)).show()
        _ = t.select(t.img.reduce(2)).show()
        _ = t.select(t.img.reduce(2, box=[0, 0, 10, 10])).show()
        _ = t.select(t.img.thumbnail([100, 100])).show()
        _ = t.select(t.img.transpose(Transpose.FLIP_LEFT_RIGHT)).show()

    def test_get_metadata(self, img_tbl: pxt.Table) -> None:
        t = img_tbl
        t.insert(img='tests/data/images/sewing-threads-smaller.jpg', category='sewing-threads', split='')
        t.add_computed_column(md=t.img.get_metadata())
        md = t.select(t.md).where(t.category == 'sewing-threads').collect()['md'][0]
        assert md == {'width': 1000, 'height': 750, 'mode': 'RGB', 'bits': 8, 'format': 'JPEG'}

        # get_metadata() returns correct type information
        expr = t.md.bits
        assert expr.col_type.is_int_type()
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="cannot resolve 'not_an_attr'"):
            _ = t.md.not_an_attr

    def test_return_types(self, uses_db: None) -> None:
        for nullable in (True, False):
            type_hint = pxt.Image[(200, 300), 'RGB']  # type: ignore
            type_hint = type_hint if nullable else pxt.Required[type_hint]  # type: ignore
            t = pxt.create_table('test', {'img': type_hint, 'info': pxt.Required[pxt.Json]}, if_exists='replace')

            assert t.img.convert(mode='L').col_type == ts.ImageType(size=(200, 300), mode='L', nullable=nullable)
            assert t.img.crop(box=(50, 50, 100, 100)).col_type == ts.ImageType(
                size=(50, 50), mode='RGB', nullable=nullable
            )
            assert t.img.crop(box=t.info).col_type == ts.ImageType(nullable=nullable)  # Non-constant box
            assert t.img.effect_spread(distance=10).col_type == ts.ImageType(
                size=(200, 300), mode='RGB', nullable=nullable
            )
            assert t.img.getchannel(channel=0).col_type == ts.ImageType(size=(200, 300), mode='L', nullable=nullable)
            assert t.img.resize(size=(100, 100)).col_type == ts.ImageType(
                size=(100, 100), mode='RGB', nullable=nullable
            )
            assert t.img.resize(size=t.info).col_type == ts.ImageType(nullable=nullable)  # Non-constant size
            assert t.img.rotate(angle=90).col_type == ts.ImageType(size=(200, 300), mode='RGB', nullable=nullable)
            assert t.img.transpose(method=Transpose.FLIP_LEFT_RIGHT).col_type == ts.ImageType(
                size=(200, 300), mode='RGB', nullable=nullable
            )

    def test_tile_iterator(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.insert(image=SAMPLE_IMAGE_URL)
        v = pxt.create_view('test_view', t, iterator=tile_iterator(t.image, (100, 100), overlap=(10, 10)))
        image: Image = t.collect()[0]['image']
        results = v.select(v.pos, v.tile, v.tile_coord, v.tile_box).order_by(v.pos).collect()
        assert image.size == (640, 480)
        assert len(results) == 42
        for j in range(6):
            for i in range(7):
                result = results[j * 7 + i]
                assert result['pos'] == j * 7 + i
                assert result['tile_coord'] == [i, j]
                box = (i * 90, j * 90, 100 + i * 90, 100 + j * 90)
                assert result['tile_box'] == list(box)
                assert result['tile'].size == (100, 100)
                tile = image.crop(box)
                assert list(result['tile'].getdata()) == list(tile.getdata())

    @pytest.mark.parametrize('overlap', [(0, 0), (10, 10)])
    @pytest.mark.parametrize('mode', ['RGB', 'L', 'RGBA'])
    def test_stitch_tiles(self, mode: str, overlap: tuple[int, int], uses_db: None) -> None:
        image_files = get_image_files()
        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.insert({'image': f} for f in image_files)
        # tiling at (100, 100) exercises edge tiles and padding on the variously sized test images
        v = pxt.create_view('test_view', t, iterator=tile_iterator(t.image, (100, 100), overlap=overlap))
        # group the tiles back per source image; selecting t.image (the grouping expr) gives each stitched
        # result's source to compare against. tiles are unstored, so the mode of the tile expression flows
        # straight into stitch_tiles; the stitched image must preserve it
        results = (
            v.select(
                t.image,
                stitched=stitch_tiles(v.pos, v.tile.convert(mode=mode), v.tile_box, v.image.width, v.image.height),
            )
            .group_by(t.image)
            .collect()
        )
        assert len(results) == len(image_files)
        for row in results:
            stitched: Image = row['stitched']
            expected = row['image'].convert(mode)
            assert stitched.mode == mode
            assert stitched.size == expected.size
            assert stitched.tobytes() == expected.tobytes()

    def test_stitch_tiles_edge_cases(self, uses_db: None) -> None:
        tiles = pxt.create_table('tiles', {'pos': pxt.Int, 'tile': pxt.Image, 'tile_box': pxt.Json, 'width': pxt.Int})
        tiles.insert(
            [
                {
                    'pos': 0,
                    'tile': PIL.Image.new('RGB', (100, 100), (255, 0, 0)),
                    'tile_box': [0, 0, 100, 100],
                    'width': 200,
                },
                {
                    'pos': 1,
                    'tile': PIL.Image.new('RGB', (100, 100), (0, 255, 0)),
                    'tile_box': [100, 0, 200, 100],
                    'width': 201,
                },
            ]
        )

        # width/height arguments that disagree within a group
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='does not match the size'):
            _ = tiles.select(stitched=stitch_tiles(tiles.pos, tiles.tile, tiles.tile_box, tiles.width, 100)).collect()

        # a 'P' tile stitches to a 'P' image with the tile's palette
        result = (
            tiles.where(tiles.width == 200)
            .select(stitched=stitch_tiles(tiles.pos, tiles.tile.quantize(), tiles.tile_box, 200, 100))
            .collect()
        )
        stitched: Image = result[0]['stitched']
        expected: Image = tiles.where(tiles.width == 200).collect()[0]['tile'].quantize()
        assert stitched.mode == 'P'
        assert stitched.getpalette() == expected.getpalette()

        # a group consisting entirely of None tiles yields None (rows with a None tile are skipped)
        nulls = pxt.create_table('null_tiles', {'pos': pxt.Int, 'tile': pxt.Image, 'tile_box': pxt.Json})
        nulls.insert(
            [
                {'pos': 0, 'tile': None, 'tile_box': [0, 0, 100, 100]},
                {'pos': 1, 'tile': None, 'tile_box': [100, 0, 200, 100]},
            ]
        )
        result = nulls.select(stitched=stitch_tiles(nulls.pos, nulls.tile, nulls.tile_box, 200, 100)).collect()
        assert len(result) == 1
        assert result[0]['stitched'] is None

    def test_tile_iterator_errors(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.insert(image=SAMPLE_IMAGE_URL)

        # Test overlap >= tile_size
        for overlap in ((0, 100), (100, 0)):
            with pxt_raises(
                pxt.ErrorCode.UNSUPPORTED_OPERATION,
                match=re.escape(
                    rf'`overlap` dimensions {list(overlap)} are not strictly smaller than `tile_size` [100, 100]'
                ),
            ):
                _ = pxt.create_view('test_view', t, iterator=tile_iterator(t.image, (100, 100), overlap=overlap))

        # Test tile_size <= 0
        for tile_size in ((0, 100), (100, 0), (-1, 100), (100, -1)):
            with pxt_raises(
                pxt.ErrorCode.INVALID_ARGUMENT,
                match=re.escape(f'`tile_size` dimensions must be positive; got {list(tile_size)}'),
            ):
                _ = pxt.create_view('test_view', t, iterator=tile_iterator(t.image, tile_size))

        # Test overlap < 0
        for overlap in ((-1, 0), (0, -1)):
            with pxt_raises(
                pxt.ErrorCode.INVALID_ARGUMENT,
                match=re.escape(f'`overlap` dimensions must be non-negative; got {list(overlap)}'),
            ):
                _ = pxt.create_view('test_view', t, iterator=tile_iterator(t.image, (100, 100), overlap=overlap))
