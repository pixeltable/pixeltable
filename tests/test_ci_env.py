"""Guards for the CI test environment (tooling compatibility, not Pixeltable behavior)."""

from pytest_benchmark.fixture import BenchmarkFixture


def test_pytest_benchmark_fixture_constructs(benchmark: BenchmarkFixture) -> None:
    """The pytest-benchmark fixture must construct.

    PXT-1714: pytest 9.2.0 removed Node._nodeid; pytest-benchmark 5.3.0 still reads it, so every
    test that requests `benchmark` errors at setup.
    """
    assert isinstance(benchmark, BenchmarkFixture)
    assert benchmark.fullname
    # One no-op round so the fixture is used; construction is what PXT-1714 broke.
    benchmark.pedantic(lambda: None, rounds=1, iterations=1)
