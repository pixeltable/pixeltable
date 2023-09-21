import datetime
import pytest

from pixeltable import catalog
from pixeltable import exceptions as exc
from pixeltable import DataFrame
from pixeltable.functions import dict_map, cast, sum, count

class TestDataFrame:
    def test_select_where(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res1 = t[t.c1, t.c2, t.c3].show(0)
        res2 = t.select(t.c1, t.c2, t.c3).show(0)
        assert res1 == res2

        res1 = t[t.c2 < 10][t.c1, t.c2, t.c3].show(0)
        res2 = t.where(t.c2 < 10).select(t.c1, t.c2, t.c3).show(0)
        assert res1 == res2

        # duplicate select list
        with pytest.raises(exc.Error) as exc_info:
            _ = t.select(t.c1).select(t.c2).show(0)
        assert 'already specified' in str(exc_info.value)

        # invalid expr in select list: Callable is not a valid literal
        with pytest.raises(TypeError) as exc_info:
            _ = t.select(datetime.datetime.now).show(0)
        assert 'Not a valid literal' in str(exc_info.value)

    def test_order_by(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res = t.select(t.c4, t.c2).order_by(t.c4).order_by(t.c2, asc=False).show(0)

        # invalid expr in order_by()
        with pytest.raises(exc.Error) as exc_info:
            _ = t.order_by(datetime.datetime.now()).show(0)
        assert 'Invalid expression' in str(exc_info.value)

    def test_count(self, test_tbl: catalog.Table, indexed_img_tbl: catalog.Table) -> None:
        t = test_tbl
        cnt = t.count()
        assert cnt == 100

        cnt = t.where(t.c2 < 10).count()
        assert cnt == 10

        # count() doesn't work with similarity search
        t = indexed_img_tbl
        probe = t.select(t.img).show(1)
        img = probe[0, 0]
        with pytest.raises(exc.Error):
            _ = t.where(t.img.nearest(img)).count()
        with pytest.raises(exc.Error):
            _ = t.where(t.img.nearest('car')).count()

        # for now, count() doesn't work with non-SQL Where clauses
        with pytest.raises(exc.Error):
            _ = t.where(t.img.width > 100).count()

    def test_select_literal(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res = t.select(1.0).where(t.c2 < 10).show(0)
        assert res.rows == [[1.0]] * 10

    def test_set_column_names(self, test_tbl : catalog.Table):
        t = test_tbl
        df : DataFrame = t.select(t.c1, t.c2, t.c3 + t.c3, t.c3+ t.c3)
        df.set_column_names({1:'c2prime', 2:'c3pc3'})
        names = df.get_column_names()
        assert names == ['c1', 'c2prime', 'c3pc3', 'col_3']

    def test_to_pytorch_dataset(self, img_tbl: catalog.Table):
        import torch
        import numpy as np
        import pickle

        t = img_tbl
        tsize = t.count()
        assert tsize > 0

        # crop images to uniform size to test tensor stacking on the loader side
        # H = 224, W = 256
        # left top right bottom
        # convert so they are all 3 channel RGB
        dfbase : DataFrame = t.select(t.img.crop((0, 0, 256, 224)).convert('RGB'), t.category, t.row_id)

        dflong = dfbase.where(t.row_id < 101)
        dflong.set_column_names({0:'imageprime'})

        long_size = dflong.count() 

        cols = dflong.get_column_names()
        
        assert long_size > 0
        assert long_size < tsize #check non-trivial selection

        dslong = dflong.to_pytorch_dataset(image_format='np')
        dslongpt = dflong.to_pytorch_dataset(image_format='pt')
        elts = []

        dataset_ids = set()
        for elt in dslong:
            assert 'imageprime' in elt
            assert 'category' in elt
            assert isinstance(elt['category'], str)

            arr = elt['imageprime']
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (224, 256, 3)
            assert arr.dtype == np.uint8
            assert len(elt) == len(cols), len(elt)
            elts.append(elt)
            
            row_id = int(elt['row_id'])
            assert row_id not in dataset_ids
            dataset_ids.add(row_id)

        # check read all rows exactly once
        assert dataset_ids == set(range(long_size))

        # Dataset must be pickleable for it to work well with multiprocessing
        x = pickle.dumps(dslong)
        _ = pickle.loads(x)

        elts =[]
        for elt in dslongpt:
            assert 'imageprime' in elt
            assert 'category' in elt
            assert isinstance(elt['category'], str)
            arr = elt['imageprime']
            assert torch.is_tensor(arr)
            assert arr.shape == (3, 224, 256)
            assert arr.dtype == torch.float32
            assert (0.0 <= arr).all()
            assert (arr <= 1.0).all()

            elts.append(elt)

        assert len(elts) == long_size

        ## now test interaction with dataloader
        def _test_loader(ds, size, **kwargs):
            dl = torch.utils.data.DataLoader(ds, **kwargs)

            loaded_ids = set()
            for b in dl:
                assert 'imageprime' in b
                assert b['imageprime'].shape[0] <= dl.batch_size
                assert torch.is_tensor(b['imageprime'])
                for row_id in b['row_id']:
                    loaded_ids.add(int(row_id))

            assert loaded_ids == set(range(size))

        _test_loader(dslongpt, size=long_size, batch_size=13, num_workers=0)
        _test_loader(dslongpt, size=long_size, batch_size=13, num_workers=1)
        _test_loader(dslongpt, size=long_size, batch_size=13, num_workers=2) # check no duplicates

        dfshort = dfbase.where(t.row_id < 3)
        dfshort.set_column_names({0:'imageprime'})
        dsshort = dfshort.to_pytorch_dataset(image_format='pt')
        _test_loader(dsshort, size=3, batch_size=13, num_workers=5) # more workers than rows

