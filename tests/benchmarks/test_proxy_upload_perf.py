import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import pixeltable as pxt

from ..utils import DatabaseRoot, get_image_files

pytestmark = pytest.mark.db_roots('cloud', reason='out-of-band media upload benchmark')


class TestProxyUploadPerformance:
    @pytest.mark.benchmark(group='proxy_image_upload')
    def test_insert_local_images(self, db_root: DatabaseRoot, benchmark: BenchmarkFixture) -> None:
        """Insert 300 local image files in one insert() call, into a new table each round."""
        row_count = 300
        rows = [{'img': path} for path in get_image_files()[:row_count]]
        tables: list[pxt.Table] = []

        def create_table() -> tuple[tuple[pxt.Table], dict]:
            t = pxt.create_table(db_root.make_catalog_path(f'upload_{len(tables)}'), {'img': pxt.Image})
            tables.append(t)
            return (t,), {}

        def insert(t: pxt.Table) -> None:
            status = t.insert(rows)
            assert status.num_rows == row_count
            assert status.num_excs == 0

        # the warmup round absorbs the connection and credential setup of a first request
        benchmark.pedantic(insert, setup=create_table, rounds=5, iterations=1, warmup_rounds=1)
        for t in tables:
            assert t.count() == row_count
