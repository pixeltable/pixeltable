"""Regression tests for xdist worker-crash reporting (PXT-1769)."""

import re

import pytest

pytest.importorskip('xdist')
pytest.importorskip('pytest_rerunfailures')


def test_worker_crash_under_loadgroup_is_a_normal_failure(pytester: pytest.Pytester) -> None:
    """A crashed xdist worker under --dist loadgroup must fail the crashed test, not INTERNALERROR.

    pytest-rerunfailures 15.1 called LoadGroupScheduling.mark_test_pending, which raises
    NotImplementedError and turns the run into exit 3. 16.1+ reports the crashed test instead.
    """
    pytester.makepyfile(
        """
        import os

        def test_crash():
            os._exit(1)

        def test_ok():
            pass
        """
    )
    result = pytester.runpytest_subprocess(
        '-n', '2', '--dist', 'loadgroup', '--reruns', '1', '-v', '-p', 'no:cacheprovider', '-o', 'addopts=', timeout=60
    )
    output = str(result.stdout) + str(result.stderr)
    assert result.ret == pytest.ExitCode.TESTS_FAILED, output
    assert 'INTERNALERROR' not in output, output
    assert 'NotImplementedError' not in output, output
    assert re.search(r'FAILED \S*::test_crash\b', output) is not None, output
