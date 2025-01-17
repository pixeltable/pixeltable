from pixeltable import udf

@udf
def duplicate_udf(n: int) -> int:
    return n + 1

@udf
def duplicate_udf(n: int) -> int:  # type: ignore[no-redef]
    return n + 2
