"""`python -m pcli.server.daemon` — pcli FastAPI daemon."""

import uvicorn

from pcli.probe import get_port
from pcli.server.app import create_app


def main() -> None:
    uvicorn.run(create_app(), host='127.0.0.1', port=get_port(), log_config=None)


if __name__ == '__main__':
    main()
