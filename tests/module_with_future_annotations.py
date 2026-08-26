from __future__ import annotations

from typing import Iterator, TypedDict

import numpy as np

import pixeltable as pxt


@pxt.udf
def future_annotations_udf(n: int) -> int:
    return n + 1


class WordRow(TypedDict):
    word: str
    position: int
    arr: pxt.Array[np.float32] | None


@pxt.iterator
def split_words(text: str) -> Iterator[WordRow]:
    for i, w in enumerate(text.split()):
        yield WordRow(word=w, position=i, arr=None)
