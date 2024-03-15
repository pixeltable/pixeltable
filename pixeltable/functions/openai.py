import base64
import io
import types
from typing import Optional

import PIL.Image
import numpy as np
from openai._types import NotGiven, NOT_GIVEN

import pixeltable as pxt
from pixeltable import env
import pixeltable.type_system as ts


@pxt.udf()
def completions_create(
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


@pxt.udf()
def assistant(
        prompt: str,
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
) -> str:
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
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
    return result.dict()['choices'][0]['message']['content']


@pxt.udf()
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


@pxt.udf()
def moderations_create(input: str, model: Optional[str] = None) -> dict:
    result = env.Env().get().openai_client.moderations.create(input=input, model=model)
    return result.dict()


# TODO(aaron-siegel): Implement batching
@pxt.udf(return_type=ts.ArrayType((None,), dtype=ts.FloatType()))
def embedding(input: str, model: str) -> np.ndarray:
    result = env.Env().get().openai_client.embeddings.create(
        input=input,
        model=model,
        encoding_format='float'
    )
    emb = result.data[0].embedding
    return np.array(emb, dtype=np.float64)
