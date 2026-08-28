import os
from typing import Callable, Iterator, TypedDict

import pytest

import pixeltable as pxt
from pixeltable import func, type_system as ts

from ..utils import DatabaseRoot, pxt_raises


@pxt.udf
def evolving_udf(n: int) -> int:
    """A UDF whose definition will change locally, but not on cloud."""
    return n + 1


class MyRow(TypedDict):
    icol: int
    scol: str


@pxt.iterator
def evolving_iterator(n: int) -> Iterator[MyRow]:
    """An iterator whose definition will change locally, but not on cloud."""
    for i in range(n):
        yield MyRow(icol=i, scol=str(i))


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

    @pxt.iterator
    def local_only_iterator(n: int) -> Iterator[MyRow]:
        for i in range(n):
            yield MyRow(icol=i, scol=str(i))

    @pxt.iterator
    def evolving_iterator_v2(a: str) -> Iterator[MyRow]:
        yield MyRow(icol=0, scol=a)

    class MyRowV3(TypedDict):
        icol: str
        scol: str

    @pxt.iterator
    def evolving_iterator_v3(n: int) -> Iterator[MyRowV3]:
        for i in range(n):
            yield MyRowV3(icol=str(i), scol=str(i))

    class MyRowV4(TypedDict):
        icol: int
        scol: str
        extra: str

    @pxt.iterator
    def evolving_iterator_v4(n: int) -> Iterator[MyRowV4]:
        for i in range(n):
            yield MyRowV4(icol=i, scol=str(i), extra=str(i))


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
    @pytest.mark.db_roots(
        'proxy', 'cloud', reason='Test is specifically intended to isolate mismatches between local and proxy/cloud.'
    )
    def test_udf_errors(self, udf_usage_type: str, db_root: DatabaseRoot) -> None:
        """Test that we get the right error message when UDF resolution fails on cloud."""
        import tests.cloud.test_cloud_errors  # noqa: PLW0406

        p = db_root.make_catalog_path

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

    @pytest.mark.db_roots(
        'proxy', 'cloud', reason='Test is specifically intended to isolate mismatches between local and proxy/cloud.'
    )
    def test_iterator_errors(self, db_root: DatabaseRoot) -> None:
        """Test that we get the right error message when iterator resolution fails on cloud."""
        import tests.cloud.test_cloud_errors  # noqa: PLW0406

        p = db_root.make_catalog_path

        t = pxt.create_table(p('test_iterator_not_on_cloud'), {'n': pxt.Int, 'a': pxt.String})

        # Refer to an iterator that's not defined on cloud
        with pxt_raises(
            pxt.ErrorCode.FUNCTION_NOT_FOUND,
            match=r'The request references the iterator `tests\.cloud\.test_cloud_errors\.local_only_iterator`, '
            r'but that iterator is not defined in the remote database\.',
        ):
            pxt.create_view(p('view'), t, iterator=local_only_iterator(t.n))

        def mimic(it: func.GeneratingFunction) -> None:
            """Monkey-patches `tests.cloud.test_cloud_errors.evolving_iterator` with the given iterator."""
            tests.cloud.test_cloud_errors.evolving_iterator = func.GeneratingFunction(
                it.decorated_callable, it.unstored_cols, 'tests.cloud.test_cloud_errors.evolving_iterator'
            )

        # Refer to an iterator whose signature differs from cloud
        with pxt_raises(
            pxt.ErrorCode.FUNCTION_NOT_FOUND,
            match=r'The request references the iterator `tests\.cloud\.test_cloud_errors\.evolving_iterator`, '
            r'but the signature of the iterator\nin the remote database does not match its local definition\.\n'
            r'Signature of the local iterator: \(pxt\.String\) -> \.\.\.\n'
            r'Signature of the remote iterator: \(n: pxt\.Int\) -> \.\.\.',
        ):
            mimic(evolving_iterator_v2)
            pxt.create_view(p('view'), t, iterator=tests.cloud.test_cloud_errors.evolving_iterator(t.a))

        # Refer to an iterator whose output schema differs from cloud
        with pxt_raises(
            pxt.ErrorCode.FUNCTION_NOT_FOUND,
            match=r'The request references the iterator `tests\.cloud\.test_cloud_errors\.evolving_iterator`, '
            r'but the output schema of the iterator\nin the remote database does not match its local definition\.',
        ):
            mimic(evolving_iterator_v3)
            pxt.create_view(p('view'), t, iterator=tests.cloud.test_cloud_errors.evolving_iterator(t.n))

        # Refer to an iterator with an output field that doesn't exist on cloud
        with pxt_raises(
            pxt.ErrorCode.FUNCTION_NOT_FOUND,
            match=r'The request references the iterator `tests\.cloud\.test_cloud_errors\.evolving_iterator`, '
            r'but the output schema of the iterator\nin the remote database does not match its local definition\.',
        ):
            mimic(evolving_iterator_v4)
            pxt.create_view(p('view'), t, iterator=tests.cloud.test_cloud_errors.evolving_iterator(t.n))
