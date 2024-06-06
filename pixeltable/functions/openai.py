import base64
import io
import pathlib
import uuid
from typing import Optional, TypeVar, Union, Callable

import PIL.Image
import numpy as np
import openai
import tenacity
from openai._types import NOT_GIVEN, NotGiven

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable import env
from pixeltable.func import Batch


@env.register_client('openai')
def _(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key)


def _openai_client() -> openai.OpenAI:
    return env.Env.get().get_client('openai')


# Exponential backoff decorator using tenacity.
# TODO(aaron-siegel): Right now this hardwires random exponential backoff with defaults suggested
# by OpenAI. Should we investigate making this more customizable in the future?
def _retry(fn: Callable) -> Callable:
    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(openai.RateLimitError),
        wait=tenacity.wait_random_exponential(multiplier=3, max=180),
        stop=tenacity.stop_after_attempt(20)
    )(fn)


#####################################
# Audio Endpoints

@pxt.udf(return_type=ts.AudioType())
@_retry
def speech(
        input: str,
        *,
        model: str,
        voice: str,
        response_format: Optional[str] = None,
        speed: Optional[float] = None
) -> str:
    content = _openai_client().audio.speech.create(
        input=input,
        model=model,
        voice=voice,
        response_format=_opt(response_format),
        speed=_opt(speed)
    )
    ext = response_format or 'mp3'
    output_filename = str(env.Env.get().tmp_dir / f"{uuid.uuid4()}.{ext}")
    content.write_to_file(output_filename)
    return output_filename


@pxt.udf(
    param_types=[ts.AudioType(), ts.StringType(), ts.StringType(nullable=True),
                 ts.StringType(nullable=True), ts.FloatType(nullable=True)]
)
@_retry
def transcriptions(
        audio: str,
        *,
        model: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None
) -> dict:
    file = pathlib.Path(audio)
    transcription = _openai_client().audio.transcriptions.create(
        file=file,
        model=model,
        language=_opt(language),
        prompt=_opt(prompt),
        temperature=_opt(temperature)
    )
    return transcription.dict()


@pxt.udf(
    param_types=[ts.AudioType(), ts.StringType(), ts.StringType(nullable=True), ts.FloatType(nullable=True)]
)
@_retry
def translations(
        audio: str,
        *,
        model: str,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None
) -> dict:
    file = pathlib.Path(audio)
    translation = _openai_client().audio.translations.create(
        file=file,
        model=model,
        prompt=_opt(prompt),
        temperature=_opt(temperature)
    )
    return translation.dict()


#####################################
# Chat Endpoints

@pxt.udf
@_retry
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
    result = _openai_client().chat.completions.create(
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
@_retry
def vision(
        prompt: str,
        image: PIL.Image.Image,
        *,
        model: str = 'gpt-4-vision-preview'
) -> str:
    # TODO(aaron-siegel): Decompose CPU/GPU ops into separate functions
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
    result = _openai_client().chat.completions.create(
        messages=messages,
        model=model
    )
    return result.choices[0].message.content


#####################################
# Embeddings Endpoints

_embedding_dimensions_cache: dict[str, int] = {
    'text-embedding-ada-002': 1536,
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
}


@pxt.udf(batch_size=32, return_type=ts.ArrayType((None,), dtype=ts.FloatType()))
@_retry
def embeddings(
        input: Batch[str],
        *,
        model: str,
        dimensions: Optional[int] = None,
        user: Optional[str] = None
) -> Batch[np.ndarray]:
    result = _openai_client().embeddings.create(
        input=input,
        model=model,
        dimensions=_opt(dimensions),
        user=_opt(user),
        encoding_format='float'
    )
    return [
        np.array(data.embedding, dtype=np.float64)
        for data in result.data
    ]


@embeddings.conditional_return_type
def _(model: str, dimensions: Optional[int] = None) -> ts.ArrayType:
    if dimensions is None:
        if model not in _embedding_dimensions_cache:
            # TODO: find some other way to retrieve a sample
            return ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False)
        dimensions = _embedding_dimensions_cache.get(model, None)
    return ts.ArrayType((dimensions,), dtype=ts.FloatType(), nullable=False)


#####################################
# Images Endpoints

@pxt.udf
@_retry
def image_generations(
        prompt: str,
        *,
        model: Optional[str] = None,
        quality: Optional[str] = None,
        size: Optional[str] = None,
        style: Optional[str] = None,
        user: Optional[str] = None
) -> PIL.Image.Image:
    # TODO(aaron-siegel): Decompose CPU/GPU ops into separate functions
    result = _openai_client().images.generate(
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


@image_generations.conditional_return_type
def _(size: Optional[str] = None) -> ts.ImageType:
    if size is None:
        return ts.ImageType(size=(1024, 1024))
    x_pos = size.find('x')
    if x_pos == -1:
        return ts.ImageType()
    try:
        width, height = int(size[:x_pos]), int(size[x_pos + 1:])
    except ValueError:
        return ts.ImageType()
    return ts.ImageType(size=(width, height))


#####################################
# Moderations Endpoints

@pxt.udf
@_retry
def moderations(
        input: str,
        *,
        model: Optional[str] = None
) -> dict:
    result = _openai_client().moderations.create(
        input=input,
        model=_opt(model)
    )
    return result.dict()


_T = TypeVar('_T')


def _opt(arg: _T) -> Union[_T, NotGiven]:
    return arg if arg is not None else NOT_GIVEN
