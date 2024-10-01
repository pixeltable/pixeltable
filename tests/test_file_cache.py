import os
from collections import OrderedDict
from pathlib import Path

import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.env import Env
from pixeltable.utils.filecache import FileCache

from .utils import get_image_files


class TestFileCache:
    def test_eviction(self, reset_db):
        # Set a very small cache size of 200 kiB for this test (the imagenette images are ~5-10 kiB each)
        fc = FileCache.get()
        test_capacity = 200 << 10
        fc.clear()
        fc.set_capacity(test_capacity)

        # Construct image URLs
        image_files = get_image_files()[:50]
        base_url = 'https://github.com/pixeltable/pixeltable/raw/main/tests/data/imagenette2-160/'
        image_urls = [base_url + Path(file).name for file in image_files]

        # Initialize a table and a dict to separately track the LRU order
        t = pxt.create_table('images', {'index': pxt.IntType(), 'image': pxt.ImageType()})
        lru_tracker: OrderedDict[int, (str, int)] = OrderedDict()  # index -> (url, size)
        expected_cache_size = 0
        expected_num_evictions = 0

        for index, (file, url) in enumerate(zip(image_files, image_urls)):
            if index % 5 == 2:  # Arbitrary points in time at which to adjust the LRU order
                # Every so often, query the (expected) eldest item in the cache to adjust the (expected) eviction order
                eldest_index = next(iter(lru_tracker.keys()))
                _ = t.where(t.index == eldest_index).select(t.image.rotate(180)).collect()
                lru_tracker.move_to_end(eldest_index)
            t.insert(index=index, image=url)
            file_info = os.stat(file)
            lru_tracker[index] = (url, file_info.st_size)
            expected_cache_size += file_info.st_size
            while expected_cache_size > test_capacity:
                _, (_, size) = lru_tracker.popitem(last=False)
                expected_cache_size -= size
                expected_num_evictions += 1
            assert fc.num_files() == len(lru_tracker)
            assert fc.total_size == expected_cache_size
            assert fc.num_evictions == expected_num_evictions

        assert fc.num_evictions > 0  # Sanity check that the test actually did something

        # Inspect the physical cache directory and ensure that its contents match the cache state
        files = list(Env.get().file_cache_dir.glob('*.JPEG'))
        assert len(files) == len(lru_tracker)
        assert sum(f.stat().st_size for f in files) == expected_cache_size

        # The file cache uses last modified time to track recency. If we order the directory contents by last
        # modified time, we should get an exact match vs. the LRU tracker.
        files.sort(key=lambda f: f.stat().st_mtime)
        assert [f.stat().st_size for f in files] == [size for _, size in lru_tracker.values()]

        # Re-insert some images and check that we get a "previously evicted" warning
        with pytest.warns(excs.PixeltableWarning, match='10 media file\\(s\\) had to be downloaded multiple times') as record:
            t.insert({'index': len(image_files) + n, 'image': image_urls[n]} for n in range(10))
        # Check that we saw the warning exactly once
        assert sum(r.category is excs.PixeltableWarning for r in record) == 1

        # Re-insert some more files and check that we get another warning (we should get one per top-level operation),
        # and that the new warning reflects cumulative session eviction stats.
        with pytest.warns(excs.PixeltableWarning, match='15 media file\\(s\\) had to be downloaded multiple times') as record:
            t.insert({'index': len(image_files) + n, 'image': image_urls[n]} for n in range(10, 15))
        # Check that we saw the warning exactly once
        assert sum(r.category is excs.PixeltableWarning for r in record) == 1
