"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Google Gemini API. In order to use them, you must
first `pip install google-generativeai` and configure your Gemini credentials, as described in
the [Working with Gemini](https://pixeltable.readme.io/docs/working-with-gemini) tutorial.
"""

from typing import Optional

import pixeltable as pxt
from pixeltable import env


@env.register_client('gemini')
def _(api_key: str) -> None:
    import google.generativeai as genai
    genai.configure(api_key=api_key)


def _ensure_loaded() -> None:
    env.Env.get().get_client('gemini')


@pxt.udf
async def generate_content(
    contents: str,
    *,
    model_name: str,
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

    __Requirements:__

    - `pip install google-generativeai`

    Args:
        contents: The input content to generate from.
        model_name: The name of the model to use.

    For details on the other parameters, see: <https://ai.google.dev/gemini-api/docs>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `gemini-1.5-flash`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(response=generate_content(tbl.prompt, model_name='gemini-1.5-flash'))
    """
    env.Env.get().require_package('google.generativeai')
    _ensure_loaded()
    import google.generativeai as genai

    model = genai.GenerativeModel(model_name=model_name)
    gc = genai.GenerationConfig(
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
    response = await model.generate_content_async(contents, generation_config=gc)
    return response.to_dict()

@generate_content.resource_pool
def _(model_name: str) -> str:
    return f'request-rate:gemini:{model_name}'
