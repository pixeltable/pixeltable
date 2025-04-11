from pixeltable import udf


@udf
def duplicate_udf(n: int) -> int:
    return n + 1


@udf  # type: ignore[no-redef]  # noqa: F811  # (this one gives our linters conniption fits)
def duplicate_udf(n: int) -> int:
    return n + 2
