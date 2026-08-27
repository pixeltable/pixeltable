import os
from typing import Callable

import pytest

import pixeltable as pxt
from pixeltable import func, type_system as ts

from ..utils import CatalogMode, pxt_raises


@pxt.udf
def evolving_udf(n: int) -> int:
    """A UDF whose definition will change locally, but not on cloud."""
    return n + 1


# These UDFs are defined only locally. By omitting them from both the cloud runtime and the proxy daemon, we enable
# local testing of error resolution via the 'proxy' tests.
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


@pxt.udf
def udf_with_import_error_on_cloud(cfg: str, n: int) -> int:
    """A UDF whose conditional_return_type raises an ImportError on cloud."""
    return n + 1


@udf_with_import_error_on_cloud.conditional_return_type
def _(cfg: str) -> ts.ColumnType:
    if os.environ.get('PIXELTABLE_PROXY_DAEMON') or os.environ.get('PXTCLOUD_DB'):
        import jabberwocky  # type: ignore[import-not-found]  # noqa: F401

    return ts.IntType()


class TestCloudErrors:
    @pytest.mark.parametrize('udf_usage_type', ['select', 'computed_column'])
    def test_udf_errors(
        self, udf_usage_type: str, make_catalog_path: Callable[[str], str], catalog_mode: CatalogMode
    ) -> None:
        """Test that we get the right error message when UDF resolution fails on cloud."""
        import tests.cloud.test_cloud_errors  # noqa: PLW0406

        if catalog_mode == 'local':
            pytest.skip("Does not run against 'local'.")

        p = make_catalog_path

        t = pxt.create_table(p('test_udf_not_on_cloud'), {'n': pxt.Int, 'a': pxt.String})
        accessor: Callable
        match udf_usage_type:
            case 'select':
                accessor = lambda **kwargs: t.select(**kwargs).collect()  # noqa: E731
            case 'computed_column':
                accessor = t.add_computed_column

        # Refer to a UDF that's not defined on cloud
        with pxt_raises(
            pxt.ErrorCode.FUNCTION_NOT_FOUND,
            match=r'The request references the UDF `tests\.cloud\.test_cloud_errors\.local_only_udf`, '
            r'but that UDF is not defined in the remote database\.',
        ):
            accessor(n_plus_1=local_only_udf(t.n))

        def mimic(fn: func.CallableFunction) -> None:
            """Monkey-patches `tests.cloud.test_cloud_errors.evolving_udf` with the given function."""
            tests.cloud.test_cloud_errors.evolving_udf = func.CallableFunction(
                fn.signatures, fn.py_fns, 'tests.cloud.test_cloud_errors.evolving_udf'
            )
            tests.cloud.test_cloud_errors.evolving_udf._conditional_return_type = fn._conditional_return_type

        # Refer to a UDF whose signature differs from cloud
        with pxt_raises(
            pxt.ErrorCode.FUNCTION_NOT_FOUND,
            match=r'The request references the UDF `tests\.cloud\.test_cloud_errors\.evolving_udf`, '
            r'but the signature of the UDF\nin the remote database does not match its local definition\.\n'
            r'Signature of the local UDF: \(pxt\.String\) -> pxt\.Int\n'
            r'Signature of the remote UDF: \(n: pxt\.Int\) -> pxt\.Int',
        ):
            mimic(evolving_udf_v2)
            accessor(n_plus_1=evolving_udf(t.a))

        # Refer to a UDF whose return type differs from cloud
        with pxt_raises(
            pxt.ErrorCode.FUNCTION_NOT_FOUND,
            match=r'The request references the UDF `tests\.cloud\.test_cloud_errors\.evolving_udf`, '
            r'but the return type of the UDF\nin the remote database does not match its local definition\.\n'
            r'Return type of the local UDF: String\n'
            r'Return type of the remote UDF: Int',
        ):
            mimic(evolving_udf_v3)
            accessor(n_plus_1=evolving_udf(t.n))

        # Refer to a UDF whose conditional_return_type raises an ImportError on cloud
        with pxt_raises(
            pxt.ErrorCode.FUNCTION_NOT_FOUND,
            match=r'A UDF call to `tests\.cloud\.test_cloud_errors\.udf_with_import_error_on_cloud` could not be '
            r'resolved because there are missing\ndependencies in the remote database:\n'
            r"No module named 'jabberwocky'",
        ):
            accessor(n_plus_1=udf_with_import_error_on_cloud('a_constant', t.n))
