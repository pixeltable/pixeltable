import schema  # noqa: F401 — initializes Pixeltable schema on import
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from routers.search import router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
