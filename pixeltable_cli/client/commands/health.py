from pixeltable_cli import models

from ..parser import Parser
from ..utils import get_request


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt health')
    ap.parse_args(argv)  # validates --help and rejects unknown args
    print(models.HealthResponse.model_validate(get_request('/api/health')).model_dump_json(indent=2))
