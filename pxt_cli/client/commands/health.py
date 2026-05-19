import json

from ..http import get
from ..parser import Parser


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli health')
    ap.parse_args(argv)  # validates --help and rejects unknown args
    print(json.dumps(get('/pcli/v0/health'), indent=2))
