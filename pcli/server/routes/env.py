import os

from fastapi import APIRouter

from pcli.models import EnvResponse

router = APIRouter()


@router.get('/pcli/v0/env', response_model=EnvResponse)
def env() -> EnvResponse:
    env_vars = {k: v for k, v in os.environ.items() if k.startswith('PIXELTABLE_')}
    return EnvResponse(
        env_vars=env_vars,
        config_file=os.environ.get('PIXELTABLE_CONFIG'),
    )
