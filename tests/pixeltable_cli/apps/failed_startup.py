"""An importable service whose startup fails, leaving a traceback in the pod log."""

import fastapi

app = fastapi.FastAPI()


@app.on_event('startup')
def fail_startup() -> None:
    raise RuntimeError('intentional startup failure')
