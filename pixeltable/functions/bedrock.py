"""
Pixeltable UDFs for AWS Bedrock AI models.
In order to use them, you must
first `pip install boto3` and configure your AWS credentials, as described in
the [Working with Bedrock](https://docs.pixeltable.com/howto/providers/working-with-bedrock) tutorial.
"""

import asyncio
import base64
import copy
import io
import json
import logging
import mimetypes
from base64 import b64encode
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Literal

import numpy as np
import PIL.Image
import puremagic

import pixeltable as pxt
from pixeltable import env, exprs, type_system as ts
from pixeltable.config import Config
from pixeltable.func import Tools
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names
from pixeltable.utils.image import to_base64
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import ObjectOps, ObjectPath, StorageObjectAddress

if TYPE_CHECKING:
    from botocore.client import BaseClient

    from pixeltable.utils.s3_store import S3Store

_logger = logging.getLogger('pixeltable')

_ASYNC_INVOCATION_TIMEOUT_SECS = 600.0

# Models that always require StartAsyncInvoke and produce video output.
_ASYNC_VIDEO_OUTPUT_MODEL_PREFIXES = frozenset({'amazon.nova-reel', 'luma.'})

# Models that require StartAsyncInvoke for audio/video inputs (detected via body['inputType']).
_ASYNC_EMBEDDING_MODEL_PREFIXES = frozenset({'twelvelabs.marengo'})


@env.register_client('bedrock')
def _(api_key: str | None = None, region_name: str | None = None) -> 'BaseClient':
    import boto3

    if api_key is not None:
        # Bedrock API Key (bearer token) authentication.
        # Create client with placeholder credentials; the actual bearer token
        # is injected into each request via an event handler.
        session = boto3.Session(
            aws_access_key_id='bedrock-api-key',
            aws_secret_access_key='bedrock-api-key',
            region_name=region_name or 'us-east-1',
        )
        client = session.client('bedrock-runtime')

        def _inject_bearer_token(request: Any, **_kwargs: Any) -> None:
            request.headers['Authorization'] = f'Bearer {api_key}'

        client.meta.events.register('before-send', _inject_bearer_token)
        return client

    kwargs: dict[str, Any] = {'service_name': 'bedrock-runtime'}
    if region_name is not None:
        kwargs['region_name'] = region_name
    return boto3.client(**kwargs)


# boto3 typing is weird; type information is dynamically defined, so the best we can do for the static checker is `Any`
def _bedrock_client() -> Any:
    return get_runtime().get_client('bedrock')


def _get_temp_location() -> str:
    """Get the bedrock.temp_location config value, raising a clear error if not set."""
    location = Config.get().get_value('temp_location', str, section='bedrock')
    if not location:
        raise pxt.Error(
            'bedrock.temp_location must be configured for async model invocation. '
            'Set the environment variable BEDROCK_TEMP_LOCATION or add temp_location to the '
            '[bedrock] section of config.toml.'
        )
    return location


def _requires_async(body: dict, model_id: str) -> bool:
    """Return True if this model and input combination requires StartAsyncInvoke."""
    if model_id.startswith(tuple(_ASYNC_VIDEO_OUTPUT_MODEL_PREFIXES)):
        return True
    if model_id.startswith(tuple(_ASYNC_EMBEDDING_MODEL_PREFIXES)):
        input_type = body.get('inputType', '')
        return input_type in ('audio', 'video')
    return False


def _is_media(v: Any) -> bool:
    """Return True if v is a PIL image or a file path (str) to a media file."""
    if isinstance(v, PIL.Image.Image):
        return True
    if not isinstance(v, str):
        return False
    mime = mimetypes.guess_type(v)[0]
    if not mime or not mime.startswith(('image/', 'video/', 'audio/')):
        return False
    try:
        return Path(v).exists()
    except (OSError, ValueError):
        return False


def _to_base64_str(media: PIL.Image.Image | str) -> str:
    if isinstance(media, PIL.Image.Image):
        return to_base64(media)
    with open(media, 'rb') as f:
        return b64encode(f.read()).decode('utf-8')


def _to_binary(media: PIL.Image.Image | str) -> bytes:
    """Convert PIL image or file path to raw bytes for the Converse API (expects bytes, not base64)."""
    if isinstance(media, PIL.Image.Image):
        buf = io.BytesIO()
        media.save(buf, format=media.format or 'JPEG')
        return buf.getvalue()
    with open(media, 'rb') as f:
        return f.read()


def _to_data_uri(media: PIL.Image.Image | str) -> str:
    if isinstance(media, PIL.Image.Image):
        mime = PIL.Image.MIME.get(media.format, 'image/jpeg')
        return f'data:{mime};base64,{_to_base64_str(media)}'
    mime = mimetypes.guess_type(media)[0] or 'application/octet-stream'
    return f'data:{mime};base64,{_to_base64_str(media)}'


#####################################
# Input conversion


# Models that expect data URIs (e.g. `data:image/jpeg;base64,<base64-encoded-image>`) for media values.
_DATA_URI_INPUT_MODEL_PREFIXES = frozenset(
    {
        'mistral.pixtral',
        'us.mistral.pixtral',
        'mistral.mistral-large-3',
        'us.mistral.mistral-large-3',
        'mistral.magistral',
        'us.mistral.magistral',
        'mistral.ministral',
        'us.mistral.ministral',
        'google.gemma-3',
        'nvidia.nemotron-nano-12b',
        'moonshot.',
        'moonshotai.',
        'qwen.qwen3-vl',
        'cohere.embed-v4',
    }
)


def _process_media_input(obj: Any, converter: Callable) -> Any:
    """Recursively walk a request body and apply model specific converter to media inputs."""
    if isinstance(obj, dict):
        return {k: _process_media_input(v, converter) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_process_media_input(v, converter) for v in obj]
    if _is_media(obj):
        return converter(obj)
    return obj


def _apply_invoke_model_request_conversions(body: dict, model_id: str) -> dict:
    """Convert all PIL images and media file paths in a request body to the encoding expected by the target model."""
    if any(model_id.startswith(p) for p in _DATA_URI_INPUT_MODEL_PREFIXES):
        return _process_media_input(body, _to_data_uri)
    return _process_media_input(body, _to_base64_str)


def _apply_converse_request_conversions(messages: list[dict]) -> list[dict]:
    # The Converse API always expects raw bytes regardless of model.
    return _process_media_input(copy.deepcopy(messages), _to_binary)


#####################################
# Output conversion


# Models whose invoke_model response contains base64-encoded media values.
# All other models return text, embeddings, or metadata with no media fields.
_IMAGE_GENERATION_MODEL_PREFIXES = frozenset(
    {'stability.', 'us.stability.', 'amazon.nova-canvas', 'amazon.titan-image-generator'}
)


def _decode_base64_images(value: str) -> str | PIL.Image.Image:
    """Attempt to decode a string as base64 image.
    Returns a PIL image, or the original string if it is not valid base64 media."""
    try:
        raw = base64.b64decode(value, validate=True)
        mime = puremagic.from_string(raw, mime=True)
    except Exception:
        return value
    if mime and mime.startswith('image/'):
        return PIL.Image.open(io.BytesIO(raw))
    return value


def _decode_invoke_model_response_images(obj: Any) -> Any:
    """Recursively walk a response dict and decode any base64 media strings.
    Video and audio are not decoded here — use invoke_model for models that produce video or audio output,
    which routes to StartAsyncInvoke automatically.
    """
    if isinstance(obj, dict):
        return {k: _decode_invoke_model_response_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_invoke_model_response_images(v) for v in obj]
    if isinstance(obj, str):
        return _decode_base64_images(obj)
    return obj


def _decode_invoke_model_response(response: dict, model_id: str) -> dict:
    """Decode base64 media fields in a response dict. Only applied to models known to return media."""
    if not any(model_id.startswith(p) for p in _IMAGE_GENERATION_MODEL_PREFIXES):
        return response
    return _decode_invoke_model_response_images(response)


# Default embedding dimensions for known models, used by conditional_return_type.
_embedding_dimensions: dict[str, int] = {
    'amazon.titan-embed-text-v1': 1536,
    'amazon.titan-embed-text-v2:0': 1024,
    'amazon.titan-embed-image-v1': 1024,
    'amazon.nova-2-multimodal-embeddings-v1:0': 3072,
    'cohere.embed-english-v3': 1024,
    'cohere.embed-multilingual-v3': 1024,
    'cohere.embed-v4:0': 1536,
}


@asynccontextmanager
async def _bedrock_async_invocation(
    model_id: str, body: dict, output_location: str, poll_interval_secs: float
) -> AsyncIterator[tuple['S3Store', StorageObjectAddress, str]]:
    """Submit a Bedrock async job, poll until completion, yield (store, soa, result_key), and
    delete staging S3 objects on exit."""
    from pixeltable.utils.s3_store import S3Store

    response = await asyncio.to_thread(
        _bedrock_client().start_async_invoke,
        modelId=model_id,
        modelInput=body,
        outputDataConfig={'s3OutputDataConfig': {'s3Uri': output_location}},
    )
    invocation_arn: str = response['invocationArn']
    invocation_id: str = invocation_arn.rsplit('/', maxsplit=1)[-1]

    elapsed = 0.0
    while True:
        await asyncio.sleep(poll_interval_secs)
        elapsed += poll_interval_secs
        if elapsed > _ASYNC_INVOCATION_TIMEOUT_SECS:
            raise pxt.Error(f'Async invocation {invocation_id} timed out after {_ASYNC_INVOCATION_TIMEOUT_SECS}s')
        job: dict[str, Any] = await asyncio.to_thread(_bedrock_client().get_async_invoke, invocationArn=invocation_arn)
        status: str = job['status']
        if status == 'Completed':
            break
        if status == 'Failed':
            raise pxt.Error(f'Async invocation {invocation_id} failed: {job.get("failureMessage", "unknown error")}')

    soa = ObjectPath.parse_object_storage_addr(output_location, allow_obj_name=False)
    store = ObjectOps.get_store(soa, allow_obj_name=False)
    if not isinstance(store, S3Store):
        raise pxt.Error('bedrock.temp_location must be an s3:// URI')

    result_prefix = f'{soa.prefix}{invocation_id}/'
    listed = await asyncio.to_thread(store.client().list_objects_v2, Bucket=store.bucket_name, Prefix=result_prefix)
    keys = [obj['Key'] for obj in listed.get('Contents', [])]
    if not keys:
        raise pxt.Error(f'No output found at {output_location}/{invocation_id} after job {invocation_id} completed')

    result_key = next((k for k in keys if not k.endswith('manifest.json')), None)
    if result_key is None:
        raise pxt.Error(f'No output file (only manifest) found for job {invocation_id}')

    try:
        yield store, soa, result_key
    finally:
        for key in keys:
            await asyncio.to_thread(store.client().delete_object, Bucket=store.bucket_name, Key=key)


async def _invoke_model_async(body: dict, model_id: str, poll_interval_secs: float) -> Any:
    """Execute a Bedrock async invocation and return the result."""
    output_location = _get_temp_location()

    async with _bedrock_async_invocation(model_id, body, output_location, poll_interval_secs) as (
        store,
        soa,
        result_key,
    ):
        result_uri = f'{soa.prefix_free_uri}{result_key}'
        content_type = mimetypes.guess_type(result_key)[0] or 'application/json'

        if content_type.startswith('video/') or content_type.startswith('audio/'):
            default_ext = '.mp4' if content_type.startswith('video/') else '.mp3'
            ext = mimetypes.guess_extension(content_type) or default_ext
            tmp_path = TempStore.create_path(extension=ext)
            await asyncio.to_thread(ObjectOps.copy_object_to_local_file, result_uri, tmp_path)
            return str(tmp_path)
        elif content_type.startswith('image/'):
            obj = await asyncio.to_thread(store.client().get_object, Bucket=store.bucket_name, Key=result_key)
            return PIL.Image.open(io.BytesIO(obj['Body'].read()))
        else:
            obj = await asyncio.to_thread(store.client().get_object, Bucket=store.bucket_name, Key=result_key)
            return json.loads(obj['Body'].read().decode('utf-8'))


@pxt.udf(is_deterministic=False)
async def invoke_model(
    body: dict,
    *,
    model_id: str,
    poll_interval_secs: float = 5.0,
    performance_config_latency: Literal['standard', 'optimized'] | None = None,
    service_tier: Literal['priority', 'default', 'flex', 'reserved'] | None = None,
) -> dict:
    """
    Invoke a Bedrock model.

    Equivalent to the AWS Bedrock `invoke_model` API endpoint, with automatic routing to
    `StartAsyncInvoke` for models that require it (e.g. video generation, audio/video embeddings).

    For additional details, see:
    <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html>

    PIL images and media file paths anywhere in the request body are converted automatically
    to the encoding expected by the target model. For image-generation models, base64-encoded
    images in the response are automatically decoded into `PIL.Image` objects.

    For models that require async invocation, `bedrock.temp_location` must be configured
    (set environment variable `BEDROCK_TEMP_LOCATION` or add `temp_location` to the `[bedrock]` section of config.toml).

    __Requirements:__

    - `pip install boto3`

    Args:
        body: The prompt and inference parameters as a dictionary.
        model_id: The model identifier to invoke.
        poll_interval_secs: For async models, seconds between status checks. Defaults to 5.0.
        performance_config_latency: Performance setting (`standard` or `optimized`).
        service_tier: Processing tier (`priority`, `default`, `flex`, or `reserved`).

    For details on the optional parameters, see:
    <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html>

    Returns:
        A dictionary containing the model response. For image-generation models,
        image fields are decoded to `PIL.Image` objects. For video-generation models,
        returns a `pxt.Video` path.

    Examples:
        Invoke Amazon Titan text embeddings:

        >>> body = {'inputText': tbl.text, 'dimensions': 512, 'normalize': True}
        >>> tbl.add_computed_column(
        ...     response=invoke_model(body, model_id='amazon.titan-embed-text-v2:0')
        ... )

        Invoke TwelveLabs Marengo with an image column:

        >>> body = {
        ...     'inputType': 'image',
        ...     'image': {'mediaSource': {'base64String': tbl.image}},
        ... }
        >>> tbl.add_computed_column(
        ...     response=invoke_model(
        ...         body, model_id='twelvelabs.marengo-embed-3-0-v1:0'
        ...     )
        ... )

        Invoke TwelveLabs Marengo with audio:

        >>> body = {
        ...     'inputType': 'audio',
        ...     'audio': {'mediaSource': {'base64String': tbl.audio}},
        ... }
        >>> tbl.add_computed_column(
        ...     response=invoke_model(
        ...         body, model_id='twelvelabs.marengo-embed-3-0-v1:0'
        ...     )
        ... )

        Invoke Anthropic Claude with an image:

        >>> body = {
        ...     'anthropic_version': 'bedrock-2023-05-31',
        ...     'max_tokens': 1024,
        ...     'messages': [
        ...         {
        ...             'role': 'user',
        ...             'content': [
        ...                 {
        ...                     'type': 'image',
        ...                     'source': {
        ...                         'type': 'base64',
        ...                         'media_type': 'image/jpeg',
        ...                         'data': tbl.image,
        ...                     },
        ...                 },
        ...                 {'type': 'text', 'text': "What's in this image?"},
        ...             ],
        ...         }
        ...     ],
        ... }
        >>> tbl.add_computed_column(
        ...     response=invoke_model(
        ...         body, model_id='anthropic.claude-3-haiku-20240307-v1:0'
        ...     )
        ... )

        Invoke Amazon Nova Lite with a video column:

        >>> body = {
        ...     'messages': [
        ...         {
        ...             'role': 'user',
        ...             'content': [
        ...                 {
        ...                     'video': {
        ...                         'format': 'mp4',
        ...                         'source': {'bytes': tbl.video},
        ...                     }
        ...                 },
        ...                 {'text': 'What happens in this video?'},
        ...             ],
        ...         }
        ...     ]
        ... }
        >>> tbl.add_computed_column(
        ...     response=invoke_model(body, model_id='amazon.nova-lite-v1:0')
        ... )

        Invoke Stability AI for image generation:

        >>> body = {
        ...     'prompt': tbl.prompt,
        ...     'mode': 'text-to-image',
        ...     'aspect_ratio': '1:1',
        ...     'output_format': 'jpeg',
        ... }
        >>> tbl.add_computed_column(
        ...     response=invoke_model(body, model_id='stability.sd3-5-large-v1:0')
        ... )
        >>> tbl.add_computed_column(image=tbl.response['images'][0])

        Invoke Amazon Nova Reel for video generation (auto-routes to async):

        >>> body = {
        ...     'taskType': 'TEXT_VIDEO',
        ...     'textToVideoParams': {'text': tbl.prompt},
        ...     'videoGenerationConfig': {'durationSeconds': 6, 'fps': 24},
        ... }
        >>> tbl.add_computed_column(
        ...     video=invoke_model(body, model_id='amazon.nova-reel-v1:1')
        ... )
    """
    body = _apply_invoke_model_request_conversions(body, model_id)

    if _requires_async(body, model_id):
        return await _invoke_model_async(body, model_id, poll_interval_secs)

    kwargs: dict[str, Any] = {
        'body': json.dumps(body),
        'modelId': model_id,
        'contentType': 'application/json',
        'accept': 'application/json',
    }
    if performance_config_latency is not None:
        kwargs['performanceConfigLatency'] = performance_config_latency
    if service_tier is not None:
        kwargs['serviceTier'] = service_tier

    response = await asyncio.to_thread(_bedrock_client().invoke_model, **kwargs)
    result = json.loads(response['body'].read())
    return _decode_invoke_model_response(result, model_id)


@invoke_model.conditional_return_type
def _(*, model_id: str) -> ts.ColumnType:
    if any(model_id.startswith(p) for p in _ASYNC_VIDEO_OUTPUT_MODEL_PREFIXES):
        return ts.VideoType()
    return ts.JsonType()


@pxt.udf(is_deterministic=False)
async def converse(
    messages: list[dict[str, Any]],
    *,
    model_id: str,
    system: list[dict[str, Any]] | None = None,
    inference_config: dict | None = None,
    additional_model_request_fields: dict | None = None,
    tool_config: list[dict] | None = None,
) -> dict:
    """
    Generate a conversation response.

    Equivalent to the AWS Bedrock `converse` API endpoint.
    For additional details, see:
    <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html>

    PIL images and media file paths in `messages[*].content[*].image|video|audio.source.bytes`
    are converted to raw bytes automatically.

    __Requirements:__

    - `pip install boto3`

    Args:
        messages: Input messages.
        model_id: The model that will complete your prompt.
        system: An optional system prompt.
        inference_config: Base inference parameters to use.
        additional_model_request_fields: Additional inference parameters to use.
        tool_config: An optional list of Pixeltable tools to use.

    For details on the optional parameters, see:
    <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `anthropic.claude-3-haiku-20240307-v1:0`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> msgs = [{'role': 'user', 'content': [{'text': tbl.prompt}]}]
        ... tbl.add_computed_column(
        ...     response=converse(
        ...         msgs, model_id='anthropic.claude-3-haiku-20240307-v1:0'
        ...     )
        ... )

        Pass an image via the Converse API:

        >>> msgs = [
        ...     {
        ...         'role': 'user',
        ...         'content': [
        ...             {'image': {'format': 'jpeg', 'source': {'bytes': tbl.image}}},
        ...             {'text': "What's in this image?"},
        ...         ],
        ...     }
        ... ]
        >>> tbl.add_computed_column(
        ...     response=converse(msgs, model_id='amazon.nova-lite-v1:0')
        ... )
    """
    messages = _apply_converse_request_conversions(messages)

    kwargs: dict[str, Any] = {'messages': messages, 'modelId': model_id}

    if system is not None:
        kwargs['system'] = system
    if inference_config is not None:
        kwargs['inferenceConfig'] = inference_config
    if additional_model_request_fields is not None:
        kwargs['additionalModelRequestFields'] = additional_model_request_fields

    if tool_config is not None:
        tool_config_ = {
            'tools': [
                {
                    'toolSpec': {
                        'name': tool['name'],
                        'description': tool['description'],
                        'inputSchema': {
                            'json': {
                                'type': 'object',
                                'properties': tool['parameters']['properties'],
                                'required': tool['required'],
                            }
                        },
                    }
                }
                for tool in tool_config
            ]
        }
        kwargs['toolConfig'] = tool_config_

    return await asyncio.to_thread(_bedrock_client().converse, **kwargs)


@pxt.udf
async def embed(text: str, *, model_id: str, dimensions: int | None = None) -> pxt.Array[(None,), np.float32]:
    """
    Generate text embeddings using Amazon Titan, Amazon Nova, or Cohere embedding models.

    Calls the AWS Bedrock `invoke_model` API for embedding models.
    For additional details, see:
    <https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html>
    <https://docs.aws.amazon.com/nova/latest/userguide/modality-embedding.html>
    <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html>

    __Requirements:__

    - `pip install boto3`

    Args:
        text: Input text to embed.
        model_id: The embedding model identifier. Supported models:
            - `amazon.titan-embed-text-v1`
            - `amazon.titan-embed-text-v2:0` (supports `dimensions`: 256, 512, 1024)
            - `amazon.nova-2-multimodal-embeddings-v1:0` (supports `dimensions`: 256, 512, 1024, 3072)
            - `cohere.embed-english-v3`
            - `cohere.embed-multilingual-v3`
            - `cohere.embed-v4:0` (supports `dimensions`: 256, 512, 1024, 1536)
        dimensions: Output embedding dimensions (model-dependent, optional).

    Returns:
        Embedding vector.

    Examples:
        Create an embedding index on a column `description` with Nova embeddings and custom dimensions:

        >>> tbl.add_embedding_index(
        ...     tbl.description,
        ...     string_embed=embed.using(
        ...         model_id='amazon.nova-2-multimodal-embeddings-v1:0',
        ...         dimensions=1024,
        ...     ),
        ... )
    """
    from botocore.exceptions import ClientError

    body: dict[str, Any]
    if 'nova' in model_id:
        body = {
            'taskType': 'SINGLE_EMBEDDING',
            'singleEmbeddingParams': {
                'embeddingPurpose': 'GENERIC_INDEX',
                'text': {'truncationMode': 'END', 'value': text},
            },
        }
        if dimensions is not None:
            body['singleEmbeddingParams']['embeddingDimension'] = dimensions
    elif model_id.startswith('cohere.embed'):
        body = {'texts': [text], 'input_type': 'search_document', 'embedding_types': ['float']}
        if dimensions is not None:
            body['output_dimension'] = dimensions
    elif 'v2' in model_id:
        body = {'inputText': text}
        if dimensions is not None:
            body['dimensions'] = dimensions
    else:
        body = {'inputText': text}

    kwargs: dict[str, Any] = {
        'body': json.dumps(body),
        'modelId': model_id,
        'contentType': 'application/json',
        'accept': 'application/json',
    }

    try:
        response = await asyncio.to_thread(_bedrock_client().invoke_model, **kwargs)
        response_body = json.loads(response['body'].read())
        if 'nova' in model_id:
            return np.array(response_body['embeddings'][0]['embedding'], dtype=np.float32)
        elif model_id.startswith('cohere.embed'):
            return np.array(response_body['embeddings']['float'][0], dtype=np.float32)
        else:
            return np.array(response_body['embedding'], dtype=np.float32)
    except ClientError as e:
        raise pxt.Error(f'Failed to generate embedding: {e}') from e


@embed.overload
async def _(image: PIL.Image.Image, *, model_id: str, dimensions: int | None = None) -> pxt.Array[(None,), np.float32]:
    """
    Generate image embeddings using Amazon Titan, Amazon Nova, or Cohere Embed v4.

    Args:
        image: Input image to embed.
        model_id: The embedding model identifier. Supported models:
            - `amazon.titan-embed-image-v1` (supports `dimensions`: 256, 384, 1024)
            - `amazon.nova-2-multimodal-embeddings-v1:0` (supports `dimensions`: 256, 512, 1024, 3072)
            - `cohere.embed-v4:0` (supports `dimensions`: 256, 512, 1024, 1536)
        dimensions: Output embedding dimensions (model-dependent, optional).

    Returns:
        Embedding vector as a float32 array.

    Examples:
        Add an embedding index on an image column using Amazon Titan:

        >>> tbl.add_embedding_index(
        ...     tbl.image,
        ...     image_embed=embed.using(model_id='amazon.titan-embed-image-v1'),
        ... )

        Add an embedding index on an image column using Amazon Nova:

        >>> tbl.add_embedding_index(
        ...     tbl.image,
        ...     image_embed=embed.using(
        ...         model_id='amazon.nova-2-multimodal-embeddings-v1:0',
        ...         dimensions=1024,
        ...     ),
        ... )
    """
    from botocore.exceptions import ClientError

    body: dict[str, Any]
    if 'nova' in model_id:
        body = {
            'taskType': 'SINGLE_EMBEDDING',
            'singleEmbeddingParams': {
                'embeddingPurpose': 'GENERIC_INDEX',
                'image': {'format': 'jpeg', 'source': {'bytes': to_base64(image, format='jpeg')}},
            },
        }
        if dimensions is not None:
            body['singleEmbeddingParams']['embeddingDimension'] = dimensions
    elif model_id.startswith('cohere.embed-v4'):
        body = {'images': [_to_data_uri(image)], 'input_type': 'search_document', 'embedding_types': ['float']}
        if dimensions is not None:
            body['output_dimension'] = dimensions
    else:
        # Titan Multimodal Embeddings G1 and Titan Text/Image Embeddings
        body = {'inputImage': to_base64(image)}
        if dimensions is not None:
            body['embeddingConfig'] = {'outputEmbeddingLength': dimensions}

    kwargs: dict[str, Any] = {
        'body': json.dumps(body),
        'modelId': model_id,
        'contentType': 'application/json',
        'accept': 'application/json',
    }

    try:
        response = await asyncio.to_thread(_bedrock_client().invoke_model, **kwargs)
        response_body = json.loads(response['body'].read())
        if 'nova' in model_id:
            return np.array(response_body['embeddings'][0]['embedding'], dtype=np.float32)
        elif model_id.startswith('cohere.embed-v4'):
            return np.array(response_body['embeddings']['float'][0], dtype=np.float32)
        else:
            return np.array(response_body['embedding'], dtype=np.float32)
    except ClientError as e:
        raise pxt.Error(f'Failed to generate embedding: {e}') from e


@embed.conditional_return_type
def _(*, model_id: str, dimensions: int | None = None) -> ts.ArrayType:
    if dimensions is not None:
        return ts.ArrayType((dimensions,), dtype=np.dtype(np.float32), nullable=False)
    if model_id in _embedding_dimensions:
        return ts.ArrayType((_embedding_dimensions[model_id],), dtype=np.dtype(np.float32), nullable=False)
    return ts.ArrayType((None,), dtype=np.dtype(np.float32), nullable=False)


def invoke_tools(tools: Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts a Bedrock response dict to Pixeltable tool invocation format and calls `tools._invoke()`."""
    return tools._invoke(_bedrock_response_to_pxt_tool_calls(response))


@pxt.udf
def _bedrock_response_to_pxt_tool_calls(response: dict) -> dict | None:
    if response.get('stopReason') != 'tool_use':
        return None

    pxt_tool_calls: dict[str, list[dict[str, Any]]] = {}
    for message in response['output']['message']['content']:
        if 'toolUse' in message:
            tool_call = message['toolUse']
            tool_name = tool_call['name']
            if tool_name not in pxt_tool_calls:
                pxt_tool_calls[tool_name] = []
            pxt_tool_calls[tool_name].append({'args': tool_call['input']})

    if len(pxt_tool_calls) == 0:
        return None

    return pxt_tool_calls


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
