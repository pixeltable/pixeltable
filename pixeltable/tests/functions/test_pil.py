from typing import cast

from PIL.Image import Transform, Quantize, Dither, Transpose

from pixeltable import Table
from pixeltable.functions.pil.image import *
from ..utils import get_image_files


class TestPil:

    def to_img(self, x: Any) -> PIL.Image:
        return x

    def test_pil(self, img_tbl: Table) -> None:
        #mask_img = next(f for f in get_image_files() if f.endswith('n03888257_1389.JPEG'))
        t = img_tbl
        _ = t[t.img.rotate(90)].show()
        _ = t[alpha_composite(t.img.convert(mode='RGBA'), t.img.rotate(90).convert(mode='RGBA'))].show()  # Needs RGBA images to work
        _ = t[blend(t.img, t.img.rotate(90), 0.5)].show()
        _ = t[composite(t.img, t.img.rotate(90), mask=t.img.convert(mode='RGBA'))].show()
        _ = t[t.img.crop([0, 0, 10, 10])].show()
        _ = t[t.img.getchannel(0)].show()
        _ = t[t.img.resize([100, 100])].show()
        _ = t[t.img.effect_spread(3)].show()
        _ = t[t.img.entropy(mask=t.img.convert(mode='L'))].show()
        _ = t[t.img.getbands()].show()
        _ = t[t.img.getbbox()].show()
        _ = t[t.img.getcolors(5)].show()
        _ = t[t.img.getextrema()].show()
        _ = t[t.img.getpalette()].show()
        _ = t[t.img.getpixel([5, 5])].show()
        _ = t[t.img.getprojection()].show()
        _ = t[t.img.histogram(mask=t.img.convert(mode='L'))].show()
        _ = t[t.img.quantize()].show()
        _ = t[t.img.quantize(256, Quantize.MEDIANCUT, 3, None, Dither.NONE)].show()
        _ = t[t.img.reduce(2)].show()
        _ = t[t.img.reduce(2, box=[0, 0, 10, 10])].show()
        _ = t[t.img.transpose(Transpose.FLIP_LEFT_RIGHT)].show()
