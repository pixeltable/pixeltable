import functools
import http.server
import os
import platform
import threading
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

import pytest

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.utils.filecache import FileCache

from .utils import get_image_files, rerun_on_network_error

pytestmark = pytest.mark.local('inspects local FileCache internals')


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def image_server() -> Iterator[str]:
    """Serve the local imagenette images over a localhost HTTP server, yielding the base URL.

    FileCache only caches external (non-file://) URLs, so exercising it needs remote-style URLs; serving them
    from localhost exercises the same download/cache path without the latency and flakiness of fetching over the
    internet.
    """
    img_dir = Path(get_image_files()[0]).parent
    handler = functools.partial(_QuietHandler, directory=str(img_dir))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}/'
    finally:
        server.shutdown()
        server.server_close()


class TestFileCache:
    # TODO: Understand why this test is flaky on Windows. (It appears to be a timing issue
    #     related to the Windows filesystem.)
    @pytest.mark.skipif(platform.system() == 'Windows', reason='Test is flaky on Windows')
    @rerun_on_network_error()
    def test_eviction(self, uses_db: None, image_server: str) -> None:
        # Set a very small cache size of 200 kiB for this test (the imagenette images are ~5-10 kiB each)
        fc = FileCache.get()
        test_capacity = 200 << 10
        fc.clear()
        fc.set_capacity(test_capacity)
        # disable the eviction lease so freshly-inserted files evict immediately, exercising pure LRU accounting
        fc.set_lease_seconds(0)

        # Construct image URLs
        image_files = get_image_files()[:50]
        image_urls = [image_server + Path(file).name for file in image_files]

        # Initialize a table and a dict to separately track the LRU order
        t = pxt.create_table('images', {'index': pxt.Int, 'image': pxt.Image})
        lru_tracker: OrderedDict[int, tuple[str, int]] = OrderedDict()  # index -> (url, size)
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
        with pytest.warns(
            pxt.PixeltableWarning, match='10 media file\\(s\\) had to be downloaded multiple times'
        ) as record:
            t.insert({'index': len(image_files) + n, 'image': image_urls[n]} for n in range(10))
        # Check that we saw the warning exactly once
        assert sum(r.category is pxt.PixeltableWarning for r in record) == 1

        # Re-insert some more files and check that we get another warning (we should get one per top-level operation),
        # and that the new warning reflects cumulative session eviction stats.
        with pytest.warns(
            pxt.PixeltableWarning, match='15 media file\\(s\\) had to be downloaded multiple times'
        ) as record:
            t.insert({'index': len(image_files) + n, 'image': image_urls[n]} for n in range(10, 15))
        # Check that we saw the warning exactly once
        assert sum(r.category is pxt.PixeltableWarning for r in record) == 1
