# ruff: noqa: RUF029
"""An importable service whose startup fails, leaving a traceback in the pod log."""

import contextlib
from typing import AsyncIterator

import fastapi


@contextlib.asynccontextmanager
async def fail_startup(app: fastapi.FastAPI) -> AsyncIterator[None]:
    raise RuntimeError('intentional startup failure')
    yield  # unreachable; makes this an async generator, as the lifespan contract requires


app = fastapi.FastAPI(lifespan=fail_startup)
