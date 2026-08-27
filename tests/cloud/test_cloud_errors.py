import os
from typing import Callable

import pytest

import pixeltable as pxt

from ..utils import CatalogMode, pxt_raises


@pxt.udf
def evolving_udf(n: int) -> int:
    """A UDF whose definition will change locally, but not on cloud."""
    return n + 1


# These UDFs are defined only locally. By omitting them from both cloud runtimes and the proxy daemon, we enable local
# testing of error resultion via the 'proxy' tests.
if not os.environ.get('PIXELTABLE_PROXY_DAEMON') and not os.environ.get('PXTCLOUD_DB'):

    @pxt.udf
    def local_only_udf(n: int) -> int:
        return n + 1

    @pxt.udf
    def evolving_udf_v2(a: str) -> int:
        return 0

    @pxt.udf
    def evolving_udf_v3(n: int) -> str:
        return ''


class TestCloudErrors:
    @pytest.mark.parametrize('udf_usage_type', ['select', 'computed_column'])
    def test_udf_errors(self, udf_usage_type: str, make_catalog_path: Callable[[str], str], catalog_mode: CatalogMode):
        """Test that we get the right error message when UDF resolution fails on cloud."""
        if catalog_mode == 'local':
            pytest.skip("Does not run against 'local'.")

        p = make_catalog_path

        t = pxt.create_table(p('test_udf_not_on_cloud'), {'n': pxt.Int})
        match udf_usage_type:
            case 'select':
                accessor = lambda **kwargs: t.select(**kwargs).collect()
            case 'computed_column':
                accessor = t.add_computed_column

        # Refer to a UDF that's not defined on cloud
        with pxt_raises(
            pxt.ErrorCode.FUNCTION_NOT_FOUND,
            match=r'The request references the UDF `tests\.cloud\.test_cloud_errors\.local_only_udf`, '
            r'but that UDF is not defined in the remote database\.',
        ):
            accessor(n_plus_1=local_only_udf(t.n))
