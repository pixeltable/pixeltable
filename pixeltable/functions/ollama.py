from typing import TYPE_CHECKING, Optional

import pixeltable as pxt
from pixeltable import env

if TYPE_CHECKING:
    import ollama


@env.register_client('ollama')
def _(host: str) -> 'ollama.Client':
    import ollama
    return ollama.Client(host=host)


def _ollama_client() -> Optional['ollama.Client']:
    try:
        return env.Env.get().get_client('ollama')
    except Exception:
        return None


@pxt.udf
def chat(
    messages: list[dict],
    *,
    model: str,
    tools: Optional[list[dict]] = None,
    format: str = '',
    options: Optional[dict] = None,
) -> dict:
    import ollama

    client = _ollama_client() or ollama
    return client.chat(
        model=model,
        messages=messages,
        tools=tools,
        format=format,
        options=options,
    )  # type: ignore[call-overload]
