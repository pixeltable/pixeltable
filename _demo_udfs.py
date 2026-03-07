"""UDFs for the multimodal demo — must live in a named module (not __main__)."""

import pixeltable as pxt


@pxt.udf
def word_count(text: str) -> int:
    return len(text.split())


@pxt.udf
def char_count(text: str) -> int:
    return len(text)


@pxt.udf
def reading_time_min(text: str) -> float:
    return round(len(text.split()) / 200, 1)


@pxt.udf
def risky_transform(text: str) -> str:
    if len(text.split()) < 40:
        raise ValueError(f'Text too short: {len(text.split())} words')
    return text.upper()


@pxt.udf
def division_example(n: int) -> float:
    if n == 0:
        raise ZeroDivisionError('cannot divide by zero')
    return 100.0 / n
