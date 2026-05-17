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


def _redact_home(value: str | None) -> str | None:
    """Replace user-specific home-directory prefixes with tokens, so output is portable
    and doesn't disclose the operator's identity (esp. when daemon output is shared)."""
    if value is None:
        return None
    pxt_home = os.environ.get('PIXELTABLE_HOME') or os.path.expanduser('~/.pixeltable')
    user_home = os.path.expanduser('~')
    # Longest-prefix first: PIXELTABLE_HOME is typically nested under $HOME.
    for prefix, token in ((pxt_home, '$PIXELTABLE_HOME'), (user_home, '$HOME')):
        if prefix and value.startswith(prefix):
            return token + value[len(prefix) :]
    return value


@router.get('/pcli/v0/env', response_model=EnvResponse)
def env() -> EnvResponse:
    reported_keys = [k for k in os.environ if k.startswith('PIXELTABLE_') or k == 'PCLI_PORT']
    env_vars: dict[str, str] = {}
    for k in reported_keys:
        raw = os.environ[k]
        if _is_sensitive(k):
            env_vars[k] = '<redacted>'
        else:
            env_vars[k] = _redact_home(raw) or raw
    credentials_present = {k: k in os.environ for k in _CREDENTIAL_VARS}
    return EnvResponse(
        env_vars=env_vars,
        config_file=_redact_home(os.environ.get('PIXELTABLE_CONFIG')),
        credentials_present=credentials_present,
    )
