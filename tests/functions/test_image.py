from PIL.Image import Dither, Quantize, Transpose

import pixeltable as pxt
from pixeltable.functions.image import alpha_composite, blend, composite


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
        _ = t.select(t.img.transpose(Transpose.FLIP_LEFT_RIGHT)).show()

    def test_return_types(self, reset_db: None) -> None:
        for nullable in (True, False):
            type_hint = pxt.Image[(200, 300), 'RGB']  # type: ignore
            type_hint = type_hint if nullable else pxt.Required[type_hint]  # type: ignore
            t = pxt.create_table('test', {'img': type_hint, 'info': pxt.Required[pxt.Json]}, if_exists='replace')

            assert t.img.convert(mode='L').col_type == pxt.ImageType(size=(200, 300), mode='L', nullable=nullable)
            assert t.img.crop(box=(50, 50, 100, 100)).col_type == pxt.ImageType(
                size=(50, 50), mode='RGB', nullable=nullable
            )
            assert t.img.crop(box=t.info).col_type == pxt.ImageType(nullable=nullable)  # Non-constant box
            assert t.img.effect_spread(distance=10).col_type == pxt.ImageType(
                size=(200, 300), mode='RGB', nullable=nullable
            )
            assert t.img.getchannel(channel=0).col_type == pxt.ImageType(size=(200, 300), mode='L', nullable=nullable)
            assert t.img.resize(size=(100, 100)).col_type == pxt.ImageType(
                size=(100, 100), mode='RGB', nullable=nullable
            )
            assert t.img.resize(size=t.info).col_type == pxt.ImageType(nullable=nullable)  # Non-constant size
            assert t.img.rotate(angle=90).col_type == pxt.ImageType(size=(200, 300), mode='RGB', nullable=nullable)
            assert t.img.transpose(method=Transpose.FLIP_LEFT_RIGHT).col_type == pxt.ImageType(
                size=(200, 300), mode='RGB', nullable=nullable
            )
