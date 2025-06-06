from typing import TYPE_CHECKING, Optional

import numpy as np

import pixeltable as pxt
from pixeltable import env
from pixeltable.func import Batch
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
def generate(
    prompt: str,
    *,
    model: str,
    suffix: str = '',
    system: str = '',
    template: str = '',
    context: Optional[list[int]] = None,
    raw: bool = False,
    format: Optional[str] = None,
    options: Optional[dict] = None,
) -> dict:
    """
    Generate a response for a given prompt with a provided model.

    Args:
        prompt: The prompt to generate a response for.
        model: The model name.
        suffix: The text after the model response.
        format: The format of the response; must be one of `'json'` or `None`.
        system: System message.
        template: Prompt template to use.
        context: The context parameter returned from a previous call to `generate()`.
        raw: If `True`, no formatting will be applied to the prompt.
        options: Additional options for the Ollama `chat` call, such as `max_tokens`, `temperature`, `top_p`, and
            `top_k`. For details, see the
            [Valid Parameters and Values](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)
            section of the Ollama documentation.
    """
    env.Env.get().require_package('ollama')
    import ollama

    client = _ollama_client() or ollama
    return client.generate(
        model=model,
        prompt=prompt,
        suffix=suffix,
        system=system,
        template=template,
        context=context,
        raw=raw,
        format=format,
        options=options,
    ).dict()  # type: ignore[call-overload]


@pxt.udf
def chat(
    messages: list[dict],
    *,
    model: str,
    tools: Optional[list[dict]] = None,
    format: Optional[str] = None,
    options: Optional[dict] = None,
) -> dict:
    """
    Generate the next message in a chat with a provided model.

    Args:
        messages: The messages of the chat.
        model: The model name.
        tools: Tools for the model to use.
        format: The format of the response; must be one of `'json'` or `None`.
        options: Additional options to pass to the `chat` call, such as `max_tokens`, `temperature`, `top_p`, and
            `top_k`. For details, see the
            [Valid Parameters and Values](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)
            section of the Ollama documentation.
    """
    env.Env.get().require_package('ollama')
    import ollama

    client = _ollama_client() or ollama
    return client.chat(model=model, messages=messages, tools=tools, format=format, options=options).dict()  # type: ignore[call-overload]


@pxt.udf(batch_size=16)
def embed(
    input: Batch[str], *, model: str, truncate: bool = True, options: Optional[dict] = None
) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Generate embeddings from a model.

    Args:
        input: The input text to generate embeddings for.
        model: The model name.
        truncate: Truncates the end of each input to fit within context length.
            Returns error if false and context length is exceeded.
        options: Additional options to pass to the `embed` call.
            For details, see the
            [Valid Parameters and Values](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)
            section of the Ollama documentation.
    """
    env.Env.get().require_package('ollama')
    import ollama

    client = _ollama_client() or ollama
    results = client.embed(model=model, input=input, truncate=truncate, options=options).dict()
    return [np.array(data, dtype=np.float64) for data in results['embeddings']]


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
