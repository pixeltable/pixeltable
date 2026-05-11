import json

from ..http import get


def run(argv: list[str]) -> None:
    print(json.dumps(get('/pcli/v0/health'), indent=2))
