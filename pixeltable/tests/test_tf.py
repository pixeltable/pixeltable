from pixeltable import catalog
from pixeltable.functions import dict_map
import pixeltable.tf


class TestTf:
    def test_basic(self, test_img_tbl: catalog.Table) -> None:
        t = test_img_tbl
        m = t[t.category].categorical_map()
        ds = pixeltable.tf.to_dataset(t[t.img, dict_map(t.category, m)])
        for row in ds:
            pass
        ds = pixeltable.tf.to_dataset(t[t.img.resize((224, 224))])
        for row in ds:
            pass
        ds = pixeltable.tf.to_dataset(t[t.img.convert('RGB').resize((224, 224))])
        for row in ds:
            pass
