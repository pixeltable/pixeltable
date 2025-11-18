from random import random


def exponential_backoff(attempt: int, base: float = 2.0, max_delay: float = 16.0) -> float:
    """Generates the retry delay using exponential backoff strategy with jitter. Attempt count starts from 0."""
    basic_delay = min(max_delay, base**attempt) / 2
    return basic_delay + random() * basic_delay
