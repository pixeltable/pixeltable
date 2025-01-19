from PIL.Image import Quantize, Transpose, Dither

from pixeltable import Table
from pixeltable.functions.image import alpha_composite, blend, composite


class TestImage:
    def test_image(self, img_tbl: Table) -> None:
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
