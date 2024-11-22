"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Google Gemini API. In order to use them, you must
first `pip install google-generativeai` and configure your Gemini credentials, as described in
the [Working with Gemini](https://pixeltable.readme.io/docs/working-with-gemini) tutorial.
"""

from typing import TYPE_CHECKING, Optional

import pixeltable as pxt
from pixeltable import env


@env.register_client('gemini')
def _(api_key: str) -> None:
    import google.generativeai as genai
    genai.configure(api_key=api_key)


def _ensure_loaded() -> None:
    env.Env.get().get_client('gemini')


@pxt.udf
def generate_content(
    contents: str,
    *,
    model_name: str,
    candidate_count: Optional[int] = None,
    stop_sequences: Optional[list[str]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    response_mime_type: Optional[str] = None,
    response_schema: Optional[dict] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    response_logprobs: Optional[bool] = None,
    logprobs: Optional[int] = None,
) -> dict:
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
        seed=seed,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        response_logprobs=response_logprobs,
        logprobs=logprobs,
    )
    response = model.generate_content(contents, generation_config=gc)
    return response.dict()
