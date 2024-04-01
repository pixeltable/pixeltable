import base64
import io
from typing import Optional

import PIL.Image
import numpy as np

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable import env
from pixeltable.func import Batch


@pxt.udf
def chat_completions(
        messages: list,
        model: str,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None
) -> dict:
    from openai._types import NOT_GIVEN
    result = env.Env.get().openai_client.chat.completions.create(
        messages=messages,
        model=model,
        frequency_penalty=frequency_penalty if frequency_penalty is not None else NOT_GIVEN,
        logit_bias=logit_bias if logit_bias is not None else NOT_GIVEN,
        max_tokens=max_tokens if max_tokens is not None else NOT_GIVEN,
        n=n if n is not None else NOT_GIVEN,
        presence_penalty=presence_penalty if presence_penalty is not None else NOT_GIVEN,
        response_format=response_format if response_format is not None else NOT_GIVEN,
        seed=seed if seed is not None else NOT_GIVEN,
        top_p=top_p if top_p is not None else NOT_GIVEN,
        temperature=temperature if temperature is not None else NOT_GIVEN
    )
    return result.dict()


@pxt.udf
def vision(
        prompt: str,
        image: PIL.Image.Image,
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


@pxt.udf
def moderations(input: str, model: Optional[str] = None) -> dict:
    result = env.Env().get().openai_client.moderations.create(input=input, model=model)
    return result.dict()


@pxt.udf(batch_size=32, return_type=ts.ArrayType((None,), dtype=ts.FloatType()))
def embeddings(input: Batch[str], *, model: str) -> Batch[np.ndarray]:
    result = env.Env().get().openai_client.embeddings.create(
        input=input,
        model=model,
        encoding_format='float'
    )
    embeddings = [
        np.array(data.embedding, dtype=np.float64)
        for data in result.data
    ]
    return embeddings
