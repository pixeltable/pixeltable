from typing import TYPE_CHECKING, Optional

import pixeltable as pxt
from pixeltable import env
from pixeltable.utils.code import local_public_names

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
    """
    Generate the next message in a chat with a provided model.

    Args:
        messages: The messages of the chat.
        model: The model name.
        tools: Tools for the model to use.
        format: The format of the response; must be one of `'json'` or `''` (the empty string).
        options: Additional options to pass to the `chat` call, such as `max_tokens`, `temperature`, `top_p`, and `top_k`.
            For details, see the
            [Valid Parameters and Values](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)
            section of the Ollama documentation.
    """
    env.Env.get().require_package('ollama')
    import ollama

    client = _ollama_client() or ollama
    return client.chat(
        model=model,
        messages=messages,
        tools=tools,
        format=format,
        options=options,
    )  # type: ignore[call-overload]


__all__ = local_public_names(__name__)


def __dir__():
    return __all__
