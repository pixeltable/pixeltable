from pixeltable import udf

@udf
def duplicate_udf(n: int) -> int:
    return n + 1

@udf  # type: ignore[no-redef]
def duplicate_udf(n: int) -> int:
    return n + 2
