from pixeltable import catalog
from pixeltable.type_system import StringType, BoolType, IntType, ImageType
from pixeltable.exprs import FunctionCall, Expr, CompoundPredicate
from pixeltable.functions import Function, dict_map
from pixeltable.functions.pil.image import blend
from pixeltable.functions.clip import encode_image
import pixeltable.tf


class TestTf:
    def test_basic(self, test_img_tbl: catalog.Table) -> None:
        t = test_img_tbl
        ds = pixeltable.tf.to_dataset(t[t.img])
        for row in ds:
            print(row)
        ds = pixeltable.tf.to_dataset(t[t.img.resize((224, 224))])
        for row in ds:
            print(row)
        ds = pixeltable.tf.to_dataset(t[t.img.convert('RGB').resize((224, 224))])
        for row in ds:
            print(row)
