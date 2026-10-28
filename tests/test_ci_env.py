from pytest_benchmark.fixture import BenchmarkFixture


def test_pytest_benchmark_fixture_constructs(benchmark: BenchmarkFixture) -> None:
    assert isinstance(benchmark, BenchmarkFixture)
    assert benchmark.fullname
    # pytest-benchmark warns when the fixture is unused
    benchmark.pedantic(lambda: None, rounds=1, iterations=1)
