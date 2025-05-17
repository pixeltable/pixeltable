"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Google Gemini API. In order to use them, you must
first `pip install google-genai` and configure your Gemini credentials, as described in
the [Working with Gemini](https://pixeltable.readme.io/docs/working-with-gemini) tutorial.
"""

from typing import TYPE_CHECKING, Optional

import pixeltable as pxt
from pixeltable import env

if TYPE_CHECKING:
    from google import genai


@env.register_client('gemini')
def _(api_key: str) -> 'genai.client.Client':
    from google import genai

    return genai.client.Client(api_key=api_key)


def _genai_client() -> 'genai.client.Client':
    return env.Env.get().get_client('gemini')


@pxt.udf(resource_pool='request-rate:gemini')
async def generate_content(
    contents: str,
    *,
    model: str,
    candidate_count: Optional[int] = None,
    stop_sequences: Optional[list[str]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    response_mime_type: Optional[str] = None,
    response_schema: Optional[dict] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
) -> dict:
    """
    Generate content from the specified model. For additional details, see:
    <https://ai.google.dev/gemini-api/docs>

    Request throttling:
    Applies the rate limit set in the config (section `gemini`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install google-genai`

    Args:
        contents: The input content to generate from.
        model: The name of the model to use.

    For details on the other parameters, see: <https://ai.google.dev/gemini-api/docs>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `gemini-1.5-flash`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(response=generate_content(tbl.prompt, model='gemini-1.5-flash'))
    """
    env.Env.get().require_package('google.genai')
    from google.genai import types

    config = types.GenerateContentConfig(
        candidate_count=candidate_count,
        stop_sequences=stop_sequences,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )

    response = await _genai_client().aio.models.generate_content(model=model, contents=contents, config=config)
    return response.model_dump()


@generate_content.resource_pool
def _(model: str) -> str:
    return f'request-rate:gemini:{model}'
