"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the TwelveLabs API. In order to use them, you must
first `pip install twelvelabs` and configure your TwelveLabs credentials, as described in
the [Working with TwelveLabs](https://pixeltable.readme.io/docs/working-with-twelvelabs) tutorial.
"""

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import pixeltable as pxt
from pixeltable import env
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from twelvelabs import AsyncTwelveLabs


@env.register_client('twelvelabs')
def _(api_key: str) -> 'AsyncTwelveLabs':
    from twelvelabs import AsyncTwelveLabs

    return AsyncTwelveLabs(api_key=api_key)


def _twelvelabs_client() -> 'AsyncTwelveLabs':
    return env.Env.get().get_client('twelvelabs')


@pxt.udf(resource_pool='request-rate:twelvelabs')
async def embed(
    model_name: str,
    *,
    text: str | None = None,
    text_truncate: Literal['none', 'start', 'end'] | None = None,
    audio: pxt.Audio | None = None,
    # TODO: support images
    # image: pxt.Image | None = None,
    **kwargs: Any,
) -> pxt.Array[(1024,), pxt.Float]:
    """
    Creates an embedding vector for the given `text`, `audio`, or `image` parameter. Only one of `text`, `audio`, or
    `image` may be specified.

    Equivalent to the TwelveLabs Embed API.
    https://docs.twelvelabs.io/v1.3/docs/guides/create-embeddings

    Request throttling:
    Applies the rate limit set in the config (section `twelvelabs`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install twelvelabs`

    Args:
        model_name: The name of the model to use. Check
            [the TwelveLabs documentation](https://docs.twelvelabs.io/v1.3/sdk-reference/python/create-text-image-and-audio-embeddings)
            for available models.
        text: The text to embed.
        text_truncate: Truncation mode for the text.
        audio: The audio to embed.

    Returns:
        The embedding.

    Examples:
        Add a computed column `embed` for an embedding of a string column `input`:

        >>> tbl.add_computed_column(
        ...     embed=embed(model_name='Marengo-retrieval-2.7', text=tbl.input)
        ... )
    """
    cl = _twelvelabs_client()
    res = await cl.embed.create(
        model_name=model_name, text=text, text_truncate=text_truncate, audio_file=audio, **kwargs
    )
    if text is not None:
        if res.text_embedding is None:
            raise pxt.Error(f"Didn't receive embedding for text: {text}")
        vector = res.text_embedding.segments[0].float_
        return np.array(vector, dtype=np.float64)
    # TODO: handle audio and image, once we know how to get a non-error response
    return None


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
