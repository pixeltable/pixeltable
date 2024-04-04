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
    result = env.Env.get().openai_client.chat.completions.create(
        messages=messages,
        model=model,
        frequency_penalty=_opt(frequency_penalty),
        logit_bias=_opt(logit_bias),
        logprobs=_opt(logprobs),
        top_logprobs=_opt(top_logprobs),
        max_tokens=_opt(max_tokens),
        n=_opt(n),
        presence_penalty=_opt(presence_penalty),
        response_format=_opt(response_format),
        seed=_opt(seed),
        stop=_opt(stop),
        temperature=_opt(temperature),
        top_p=_opt(top_p),
        tools=_opt(tools),
        tool_choice=_opt(tool_choice),
        user=_opt(user)
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
        user=_opt(user),
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
        model=_opt(model)
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
    b64_bytes = base64.b64decode(b64_str)
    img = PIL.Image.open(io.BytesIO(b64_bytes))
    img.load()
    return img


_T = TypeVar('_T')


def _opt(arg: _T) -> Union[_T, NotGiven]:
    return arg if arg is not None else NOT_GIVEN
