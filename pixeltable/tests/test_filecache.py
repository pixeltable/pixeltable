from pixeltable import catalog
from pixeltable.utils.filecache import FileCache


class TestFileCache:
    def test_mru(self, img_tbl: catalog.Table) -> None:
        t = img_tbl
        # add a cached column
        t.add_column(catalog.Column('c1', computed_with=t.img.rotate(90)))
        # shrink the file cache to a size that forces evictions
        FileCache.get().clear(capacity=64*1024)
        # first query: nothing has been cached, nothing gets evicted
        _ = t[t.c1].show(0)
        cache_stats1 = FileCache.get().stats()
        assert cache_stats1.num_hits == 0
        assert cache_stats1.num_evictions == 0
        # second query: some fraction has been cached, nothing gets evicted
        _ = t[t.c1].show(0)
        cache_stats2 = FileCache.get().stats()
        assert cache_stats2.num_evictions == 0
        assert cache_stats2.num_hits > 0
        # third query: the same fraction has been cached, nothing gets evicted
        _ = t[t.c1].show(0)
        cache_stats3 = FileCache.get().stats()
        assert cache_stats3.num_evictions == 0
        assert cache_stats3.num_hits == cache_stats2.num_hits * 2
