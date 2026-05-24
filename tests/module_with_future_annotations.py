from __future__ import annotations

import pixeltable as pxt


@pxt.udf
def future_annotations_udf(n: int) -> int:
    return n + 1
