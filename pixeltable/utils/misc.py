from typing import Any


def non_none_dict_factory(d: list[tuple[str, Any]]) -> dict:
    return {k: v for (k, v) in d if v is not None}
