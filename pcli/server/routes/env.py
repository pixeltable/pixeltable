import os

from fastapi import APIRouter

from pcli.models import EnvResponse

router = APIRouter()

# Vars reported value-redacted. Match by suffix to catch provider-specific names
# (e.g. PIXELTABLE_DB_CONNECT_STR, plus any *_API_KEY / *_TOKEN / *_SECRET / *_PASSWORD).
_SENSITIVE_NAMES = {'PIXELTABLE_DB_CONNECT_STR'}
_SENSITIVE_SUFFIXES = ('_API_KEY', '_TOKEN', '_SECRET', '_PASSWORD')

# Common credential vars reported as presence-only (true/false) regardless of prefix.
_CREDENTIAL_VARS = (
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY',
    'GEMINI_API_KEY',
    'GOOGLE_API_KEY',
    'MISTRAL_API_KEY',
    'GROQ_API_KEY',
    'COHERE_API_KEY',
    'TOGETHER_API_KEY',
    'HF_TOKEN',
    'HUGGINGFACE_API_KEY',
    'REPLICATE_API_TOKEN',
    'FIREWORKS_API_KEY',
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
)


def _is_sensitive(name: str) -> bool:
    return name in _SENSITIVE_NAMES or any(name.endswith(s) for s in _SENSITIVE_SUFFIXES)


@router.get('/pcli/v0/env', response_model=EnvResponse)
def env() -> EnvResponse:
    reported_keys = [k for k in os.environ if k.startswith('PIXELTABLE_') or k == 'PCLI_PORT']
    env_vars = {k: ('<redacted>' if _is_sensitive(k) else os.environ[k]) for k in reported_keys}
    credentials_present = {k: k in os.environ for k in _CREDENTIAL_VARS}
    return EnvResponse(
        env_vars=env_vars, config_file=os.environ.get('PIXELTABLE_CONFIG'), credentials_present=credentials_present
    )
