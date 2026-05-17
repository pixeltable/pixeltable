import os

from fastapi import APIRouter

from pcli._paths import redact_home
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


def _redact_user_home(value: str) -> str:
    """Layer $HOME redaction on top of the shared $PIXELTABLE_HOME redaction.

    PIXELTABLE_HOME is typically nested under $HOME, so we run redact_home() first; a
    remaining absolute path inside $HOME (e.g. a custom file_cache_dir outside the pxt
    home) then becomes `$HOME/...`. Both sides go through realpath so symlinked layouts
    (macOS `/private/Users/me`) match.
    """
    after_pxt = redact_home(value) or value
    if after_pxt.startswith('$PIXELTABLE_HOME'):
        return after_pxt
    try:
        user_home = os.path.realpath(os.path.expanduser('~'))
        target = os.path.realpath(after_pxt)
    except OSError:
        return after_pxt
    if target == user_home:
        return '$HOME'
    if target.startswith(user_home + os.sep):
        return '$HOME' + target[len(user_home) :]
    return after_pxt


@router.get('/pcli/v0/env', response_model=EnvResponse)
def env() -> EnvResponse:
    reported_keys = [k for k in os.environ if k.startswith('PIXELTABLE_') or k == 'PCLI_PORT']
    env_vars: dict[str, str] = {}
    for k in reported_keys:
        raw = os.environ[k]
        env_vars[k] = '<redacted>' if _is_sensitive(k) else _redact_user_home(raw)
    credentials_present = {k: k in os.environ for k in _CREDENTIAL_VARS}
    config_file = os.environ.get('PIXELTABLE_CONFIG')
    return EnvResponse(
        env_vars=env_vars,
        config_file=_redact_user_home(config_file) if config_file is not None else None,
        credentials_present=credentials_present,
    )
