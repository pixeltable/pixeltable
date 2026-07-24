import json

from ..parser import Parser
from ..utils import get_request


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt health')
    ap.parse_args(argv)  # validates --help and rejects unknown args
    print(json.dumps(get_request('/api/health'), indent=2))
