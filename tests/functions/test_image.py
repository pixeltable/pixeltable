import pytest
from PIL.Image import Dither, Image, Quantize, Transpose

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.functions.image import (
    alpha_composite,
    blend,
    composite,
    expand_bbox,
    fit_bbox_to_aspect,
    offset_bbox,
    rescale_bbox,
    tile_iterator,
)

from ..utils import SAMPLE_IMAGE_URL


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
        results = v.select(v.tile, v.tile_coord, v.tile_box).order_by(v.pos).collect()
        assert image.size == (640, 480)
        assert len(results) == 42
        for j in range(6):
            for i in range(7):
                result = results[j * 7 + i]
                assert result['tile_coord'] == [i, j]
                box = (i * 90, j * 90, 100 + i * 90, 100 + j * 90)
                assert result['tile_box'] == list(box)
                assert result['tile'].size == (100, 100)
                tile = image.crop(box)
                assert list(result['tile'].getdata()) == list(tile.getdata())

    # =========================================================================
    # PXT-968: expand_bbox
    # =========================================================================

    def test_expand_bbox_margin_factor(self, uses_db: None) -> None:
        t = pxt.create_table('test_expand', {'image': pxt.Image, 'bbox': pxt.Json})
        t.insert(image=SAMPLE_IMAGE_URL, bbox=[100, 100, 200, 200])
        t.add_computed_column(padded=expand_bbox(t.bbox, t.image.width, t.image.height, margin_factor=1.5))
        row = t.select(t.padded).collect()[0]
        x1, y1, x2, y2 = row['padded']
        # Original 100x100 box centred at (150,150), expanded by 1.5x → 150x150
        assert x2 - x1 == 150
        assert y2 - y1 == 150
        # Centre should remain at (150, 150)
        assert (x1 + x2) / 2 == 150
        assert (y1 + y2) / 2 == 150

    def test_expand_bbox_padding(self, uses_db: None) -> None:
        t = pxt.create_table('test_expand_pad', {'image': pxt.Image, 'bbox': pxt.Json})
        t.insert(image=SAMPLE_IMAGE_URL, bbox=[100, 100, 200, 200])
        t.add_computed_column(padded=expand_bbox(t.bbox, t.image.width, t.image.height, padding=20))
        row = t.select(t.padded).collect()[0]
        x1, y1, x2, y2 = row['padded']
        assert (x1, y1, x2, y2) == (80, 80, 220, 220)

    def test_expand_bbox_clamps_to_bounds(self, uses_db: None) -> None:
        t = pxt.create_table('test_expand_clamp', {'image': pxt.Image, 'bbox': pxt.Json})
        # SAMPLE_IMAGE_URL is 640x480
        t.insert(image=SAMPLE_IMAGE_URL, bbox=[0, 0, 50, 50])
        t.add_computed_column(padded=expand_bbox(t.bbox, t.image.width, t.image.height, padding=100))
        row = t.select(t.padded).collect()[0]
        x1, y1, _x2, _y2 = row['padded']
        assert x1 == 0
        assert y1 == 0

    # =========================================================================
    # PXT-967: rescale_bbox / offset_bbox
    # =========================================================================

    def test_rescale_bbox(self, uses_db: None) -> None:
        t = pxt.create_table('test_rescale', {'bbox': pxt.Json})
        t.insert(bbox=[100, 100, 200, 200])
        t.add_computed_column(scaled=rescale_bbox(t.bbox, (1000, 1000), (500, 500)))
        row = t.select(t.scaled).collect()[0]
        assert row['scaled'] == [50, 50, 100, 100]

    def test_offset_bbox(self, uses_db: None) -> None:
        t = pxt.create_table('test_offset', {'bbox': pxt.Json, 'crop': pxt.Json})
        t.insert(bbox=[150, 150, 250, 250], crop=[100, 100, 300, 300])
        t.add_computed_column(offset=offset_bbox(t.bbox, t.crop))
        row = t.select(t.offset).collect()[0]
        assert row['offset'] == [50, 50, 150, 150]

    def test_offset_bbox_outside_returns_none(self, uses_db: None) -> None:
        t = pxt.create_table('test_offset_none', {'bbox': pxt.Json, 'crop': pxt.Json})
        t.insert(bbox=[500, 500, 600, 600], crop=[0, 0, 100, 100])
        t.add_computed_column(offset=offset_bbox(t.bbox, t.crop))
        row = t.select(t.offset).collect()[0]
        assert row['offset'] is None

    # =========================================================================
    # PXT-966: fit_bbox_to_aspect
    # =========================================================================

    def test_fit_bbox_to_aspect_portrait(self, uses_db: None) -> None:
        t = pxt.create_table('test_fit', {'bbox': pxt.Json})
        # 1920x1080 frame, subject at centre
        t.insert(bbox=[860, 440, 1060, 640])
        t.add_computed_column(crop_box=fit_bbox_to_aspect(t.bbox, 1920, 1080, aspect_ratio='9:16'))
        row = t.select(t.crop_box).collect()[0]
        x1, y1, x2, y2 = row['crop_box']
        w = x2 - x1
        h = y2 - y1
        # The crop should match 9:16 aspect ratio (tolerance for rounding)
        assert abs(w / h - 9 / 16) < 0.02

    def test_fit_bbox_to_aspect_landscape(self, uses_db: None) -> None:
        t = pxt.create_table('test_fit_land', {'bbox': pxt.Json})
        t.insert(bbox=[100, 100, 200, 200])
        t.add_computed_column(crop_box=fit_bbox_to_aspect(t.bbox, 1920, 1080, aspect_ratio='16:9'))
        row = t.select(t.crop_box).collect()[0]
        x1, y1, x2, y2 = row['crop_box']
        w = x2 - x1
        h = y2 - y1
        assert abs(w / h - 16 / 9) < 0.02

    def test_fit_bbox_to_aspect_square(self, uses_db: None) -> None:
        t = pxt.create_table('test_fit_sq', {'bbox': pxt.Json})
        t.insert(bbox=[100, 100, 300, 200])
        t.add_computed_column(crop_box=fit_bbox_to_aspect(t.bbox, 1920, 1080, aspect_ratio='1:1'))
        row = t.select(t.crop_box).collect()[0]
        x1, y1, x2, y2 = row['crop_box']
        w = x2 - x1
        h = y2 - y1
        assert abs(w - h) <= 1  # square

    def test_tile_iterator_errors(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.insert(image=SAMPLE_IMAGE_URL)
        for overlap in ((0, 100), (100, 0)):
            with pytest.raises(pxt.Error) as exc_info:
                _ = pxt.create_view('test_view', t, iterator=tile_iterator(t.image, (100, 100), overlap=overlap))
            assert f'overlap dimensions {list(overlap)} are not strictly smaller than tile size [100, 100]' in str(
                exc_info.value
            )
