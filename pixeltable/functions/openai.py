import base64
import io
from typing import Optional, TypeVar, Union

import PIL.Image
import numpy as np
from openai._types import NOT_GIVEN, NotGiven

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable import env
from pixeltable.func import Batch


@pxt.udf
def chat_completions(
        messages: list,
        *,
        model: str,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        stop: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[dict] = None,
        user: Optional[str] = None
) -> dict:
    from openai._types import NOT_GIVEN
    result = env.Env.get().openai_client.chat.completions.create(
        messages=messages,
        model=model,
        frequency_penalty=frequency_penalty if frequency_penalty is not None else NOT_GIVEN,
        logit_bias=logit_bias if logit_bias is not None else NOT_GIVEN,
        logprobs=logprobs if logprobs is not None else NOT_GIVEN,
        top_logprobs=top_logprobs if top_logprobs is not None else NOT_GIVEN,
        max_tokens=max_tokens if max_tokens is not None else NOT_GIVEN,
        n=n if n is not None else NOT_GIVEN,
        presence_penalty=presence_penalty if presence_penalty is not None else NOT_GIVEN,
        response_format=response_format if response_format is not None else NOT_GIVEN,
        seed=seed if seed is not None else NOT_GIVEN,
        stop=stop if stop is not None else NOT_GIVEN,
        temperature=temperature if temperature is not None else NOT_GIVEN,
        top_p=top_p if top_p is not None else NOT_GIVEN,
        tools=tools if tools is not None else NOT_GIVEN,
        tool_choice=tool_choice if tool_choice is not None else NOT_GIVEN,
        user=user if user is not None else NOT_GIVEN
    )
    return result.dict()


@pxt.udf
def vision(
        prompt: str,
        image: PIL.Image.Image,
        *,
        model: str = 'gpt-4-vision-preview'
) -> str:
    bytes_arr = io.BytesIO()
    image.save(bytes_arr, format='png')
    b64_bytes = base64.b64encode(bytes_arr.getvalue())
    b64_encoded_image = b64_bytes.decode('utf-8')
    messages = [
        {'role': 'user',
         'content': [
             {'type': 'text', 'text': prompt},
             {'type': 'image_url', 'image_url': {
                 'url': f'data:image/png;base64,{b64_encoded_image}'
             }}
         ]}
    ]
    result = env.Env.get().openai_client.chat.completions.create(
        messages=messages,
        model=model
    )
    return result.choices[0].message.content


@pxt.udf(batch_size=32, return_type=ts.ArrayType((None,), dtype=ts.FloatType()))
def embeddings(
        input: Batch[str],
        *,
        model: str,
        user: Optional[str] = None
) -> Batch[np.ndarray]:
    result = env.Env().get().openai_client.embeddings.create(
        input=input,
        model=model,
        user=user if user is not None else NOT_GIVEN,
        encoding_format='float'
    )
    embeddings = [
        np.array(data.embedding, dtype=np.float64)
        for data in result.data
    ]
    return embeddings


@pxt.udf
def moderations(
        input: str,
        *,
        model: Optional[str] = None
) -> dict:
    result = env.Env().get().openai_client.moderations.create(
        input=input,
        model=model if model is not None else NOT_GIVEN
    )
    return result.dict()


@pxt.udf
def image_generations(
        prompt: str,
        *,
        model: Optional[str] = None,
        quality: Optional[str] = None,
        size: Optional[str] = None,
        style: Optional[str] = None,
        user: Optional[str] = None
) -> PIL.Image.Image:
    result = env.Env.get().openai_client.images.generate(
        prompt=prompt,
        model=_opt(model),
        quality=_opt(quality),
        size=_opt(size),
        style=_opt(style),
        user=_opt(user),
        response_format="b64_json"
    )
    b64_str = result.data[0].b64_json
    b64_bytes = b64_str.encode('utf-8')
    return PIL.Image.open(io.BytesIO(b64_bytes))


_T = TypeVar('_T')


def _opt(arg: _T) -> Union[_T, NotGiven]:
    return arg if arg is not None else NOT_GIVEN
