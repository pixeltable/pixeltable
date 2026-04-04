"""
Pixeltable UDFs for AWS Bedrock AI models.

Provides access to AWS Bedrock foundation models via four APIs:
- `invoke_model`: raw passthrough to the Bedrock InvokeModel API, supports all models
- `invoke_model_async`: asynchronous invocation via StartAsyncInvoke for models that require it
- `converse`: unified conversational API, supports all text/vision models that have Converse support
- `embed`: typed embedding helper for Amazon Titan, Amazon Nova, and Cohere embedding models

PIL images and file paths (for audio/video) passed in request bodies are converted to the
appropriate base64 encoding automatically based on the target model's schema.
"""

import asyncio
import copy
import json
import logging
from base64 import b64encode
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
import PIL.Image

import pixeltable as pxt
from pixeltable import env, exprs, type_system as ts
from pixeltable.func import Tools
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names
from pixeltable.utils.image import to_base64
from pixeltable.utils.object_stores import ObjectPath, ObjectOps
from pixeltable.utils.s3_store import S3Store

if TYPE_CHECKING:
    from botocore.client import BaseClient

_logger = logging.getLogger('pixeltable')


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


def _bedrock_client() -> Any:
    return get_runtime().get_client('bedrock')


def _is_media(v: Any) -> bool:
    return isinstance(v, (PIL.Image.Image, str)) and not isinstance(v, bool)


def _to_base64_str(media: PIL.Image.Image | str) -> str:
    if isinstance(media, PIL.Image.Image):
        return to_base64(media)
    with open(media, 'rb') as f:
        return b64encode(f.read()).decode('utf-8')


def _to_binary(media: PIL.Image.Image | str) -> bytes:
    """Convert PIL image or file path to raw bytes for the Converse API (expects bytes, not base64)."""
    if isinstance(media, PIL.Image.Image):
        import io

        buf = io.BytesIO()
        media.save(buf, format='JPEG')
        return buf.getvalue()
    with open(media, 'rb') as f:
        return f.read()


def _to_data_uri(media: PIL.Image.Image | str) -> str:
    return f'data:image/jpeg;base64,{_to_base64_str(media)}'


# Media field paths for models whose bodies have media at known exact paths (no array iteration).
# Keys are model ID prefixes. Supported models and their media fields:
#   twelvelabs.marengo  (e.g. twelvelabs.marengo-embed-3-0-v1:0):
#     image|audio|video|text_image -> mediaSource -> base64String
#   twelvelabs.pegasus  (e.g. twelvelabs.pegasus-1-2-v1:0):
#     mediaSource -> base64String  (flat schema, no inputType nesting)
#   amazon.titan-embed-image  (e.g. amazon.titan-embed-image-v1):
#     inputImage
#   amazon.nova-canvas  (e.g. amazon.nova-canvas-v1:0):
#     textToImageParams.conditionImage, inPaintingParams.image, outPaintingParams.image,
#     imageVariationParams.images, colorGuidedGenerationParams.referenceImage,
#     imageConditioningParams.conditionImage
#   amazon.titan-image-generator  (e.g. amazon.titan-image-generator-v2:0):
#     imageGenerationConfig.conditionImage, inPaintingParams.image, outPaintingParams.image
#   stability.  (e.g. stability.sd3-5-large-v1:0, stability.stable-image-core-v1:1):
#     init_image
#   us.stability.  (e.g. us.stability.sd3-5-large-v1:0):
#     init_image
#   amazon.nova-2-multimodal-embeddings  (e.g. amazon.nova-2-multimodal-embeddings-v1:0):
#     singleEmbeddingParams -> image|audio|video -> source -> bytes  (sync path, files < 25 MB)
_DIRECT_MEDIA_PATHS: dict[str, list[tuple[list[str], Callable]]] = {
    # Marengo: media nested under inputType key (e.g. image.mediaSource.base64String)
    'twelvelabs.marengo': [
        (['image', 'mediaSource', 'base64String'], _to_base64_str),
        (['audio', 'mediaSource', 'base64String'], _to_base64_str),
        (['video', 'mediaSource', 'base64String'], _to_base64_str),
        (['text_image', 'mediaSource', 'base64String'], _to_base64_str),
    ],
    # Pegasus: flat schema, mediaSource at top level
    'twelvelabs.pegasus': [(['mediaSource', 'base64String'], _to_base64_str)],
    'amazon.titan-embed-image': [(['inputImage'], _to_base64_str)],
    'amazon.nova-canvas': [
        (['textToImageParams', 'conditionImage'], _to_base64_str),
        (['inPaintingParams', 'image'], _to_base64_str),
        (['outPaintingParams', 'image'], _to_base64_str),
        (['imageVariationParams', 'images'], _to_base64_str),
        (['colorGuidedGenerationParams', 'referenceImage'], _to_base64_str),
        (['imageConditioningParams', 'conditionImage'], _to_base64_str),
    ],
    'amazon.titan-image-generator': [
        (['imageGenerationConfig', 'conditionImage'], _to_base64_str),
        (['inPaintingParams', 'image'], _to_base64_str),
        (['outPaintingParams', 'image'], _to_base64_str),
    ],
    'stability.': [(['init_image'], _to_base64_str)],
    # Cross-region inference profile variant (e.g. us.stability.sd3-5-large-v1:0)
    'us.stability.': [(['init_image'], _to_base64_str)],
    'amazon.nova-2-multimodal-embeddings': [
        (['singleEmbeddingParams', 'image', 'source', 'bytes'], _to_base64_str),
        (['singleEmbeddingParams', 'audio', 'source', 'bytes'], _to_base64_str),
        (['singleEmbeddingParams', 'video', 'source', 'bytes'], _to_base64_str),
    ],
}


def _apply_direct_conversions(body: dict, model_id: str) -> dict:
    rules = next((r for p, r in _DIRECT_MEDIA_PATHS.items() if model_id.startswith(p)), [])
    if not rules:
        return body
    body = copy.deepcopy(body)
    for keys, converter in rules:
        node: Any = body
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        if isinstance(node, dict):
            leaf = keys[-1]
            if leaf in node and _is_media(node[leaf]):
                node[leaf] = converter(node[leaf])
    return body


# Amazon Nova models (nova-lite-v1:0, nova-pro-v1:0, nova-premier-v1:0, nova-2-lite-v1:0) use
#   messages[*].content[*].<image|video|audio>.source.bytes
#
# Anthropic Claude models (claude-3-haiku-20240307-v1:0, claude-sonnet-4-20250514-v1:0, ...) use
#   messages[*].content[*].source.data  (when block type == 'image')
#
# OpenAI-compatible models (mistral.pixtral-large-2502-v1:0, mistral.mistral-large-3-675b-instruct,
#   google.gemma-3-12b-it, nvidia.nemotron-nano-12b-v2, moonshotai.kimi-k2.5,
#   qwen.qwen3-vl-235b-a22b, cohere.embed-v4:0) use
#   messages[*].content[*].image_url.url  encoded as a data URI
#
#   Cohere embed-v4 additionally uses an inputs[] array (same image_url schema):
#   inputs[*].content[*].image_url.url  encoded as a data URI
#
# Meta Llama models (llama3-2-11b-instruct-v1:0, llama4-maverick-17b-instruct-v1:0) use
#   body.images[*]
# Note: cross-region inference profile IDs carry a leading "us." (e.g. us.meta.llama3-2-11b-instruct-v1:0);
#       both the bare and prefixed forms are matched.
_MESSAGES_BYTES_PREFIXES = frozenset(
    {'amazon.nova-lite', 'amazon.nova-pro', 'amazon.nova-premier', 'amazon.nova-2-lite'}
)

_MESSAGES_ANTHROPIC_PREFIXES = frozenset({'anthropic.'})

_MESSAGES_IMAGE_URL_PREFIXES = frozenset(
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

_TOP_LEVEL_IMAGES_PREFIXES = frozenset({'meta.llama3-2', 'us.meta.llama3-2', 'meta.llama4', 'us.meta.llama4'})


def _apply_image_url_conversions_in_content(content_list: list) -> None:
    """Convert image_url media values to data URIs within a content block list (in-place)."""
    for block in content_list:
        if not isinstance(block, dict):
            continue
        img_url = block.get('image_url')
        if isinstance(img_url, dict):
            if 'url' in img_url and _is_media(img_url['url']):
                img_url['url'] = _to_data_uri(img_url['url'])
        elif _is_media(img_url):
            block['image_url'] = _to_data_uri(img_url)


def _apply_recursive_conversions(body: dict, model_id: str) -> dict:
    use_bytes = any(model_id.startswith(p) for p in _MESSAGES_BYTES_PREFIXES)
    use_anthropic = any(model_id.startswith(p) for p in _MESSAGES_ANTHROPIC_PREFIXES)
    use_image_url = any(model_id.startswith(p) for p in _MESSAGES_IMAGE_URL_PREFIXES)
    use_top_level_images = any(model_id.startswith(p) for p in _TOP_LEVEL_IMAGES_PREFIXES)

    if not any([use_bytes, use_anthropic, use_image_url, use_top_level_images]):
        return body

    body = copy.deepcopy(body)

    if use_bytes:
        for msg in body.get('messages', []):
            for block in msg.get('content', []):
                if not isinstance(block, dict):
                    continue
                for media_key in ('image', 'video', 'audio'):
                    src = block.get(media_key, {}).get('source', {})
                    if isinstance(src, dict) and 'bytes' in src and _is_media(src['bytes']):
                        src['bytes'] = _to_base64_str(src['bytes'])

    if use_anthropic:
        for msg in body.get('messages', []):
            for block in msg.get('content', []):
                if not isinstance(block, dict):
                    continue
                if block.get('type') == 'image':
                    src = block.get('source', {})
                    if isinstance(src, dict) and 'data' in src and _is_media(src['data']):
                        src['data'] = _to_base64_str(src['data'])

    if use_image_url:
        # messages[*].content[*] — standard OpenAI-compatible chat path
        for msg in body.get('messages', []):
            _apply_image_url_conversions_in_content(msg.get('content', []))
        # inputs[*].content[*] — Cohere embed-v4 uses "inputs" instead of "messages"
        for item in body.get('inputs', []):
            if isinstance(item, dict):
                _apply_image_url_conversions_in_content(item.get('content', []))

    if use_top_level_images:
        images = body.get('images', [])
        for i, img in enumerate(images):
            if _is_media(img):
                images[i] = _to_base64_str(img)

    return body


def _apply_converse_conversions(messages: list[dict]) -> list[dict]:
    # The Converse API uses a single unified schema across all models:
    # messages[*].content[*].image|video|audio -> source -> bytes
    messages = copy.deepcopy(messages)
    for msg in messages:
        for block in msg.get('content', []):
            if not isinstance(block, dict):
                continue
            for media_key in ('image', 'video', 'audio'):
                src = block.get(media_key, {}).get('source', {})
                if isinstance(src, dict) and 'bytes' in src and _is_media(src['bytes']):
                    src['bytes'] = _to_binary(src['bytes'])
    return messages


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


@pxt.udf(is_deterministic=False)
async def invoke_model(
    body: dict,
    *,
    model_id: str,
    performance_config_latency: Literal['standard', 'optimized'] | None = None,
    service_tier: Literal['priority', 'default', 'flex', 'reserved'] | None = None,
) -> dict:
    """
    Invoke a Bedrock model with a raw request body.

    Equivalent to the AWS Bedrock `invoke_model` API endpoint.
    For additional details, see:
    <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html>

    PIL images and file paths (for audio/video) anywhere in the request body are
    converted automatically to the base64 encoding expected by the target model.

    __Requirements:__

    - `pip install boto3`

    Args:
        body: The prompt and inference parameters as a dictionary.
        model_id: The model identifier to invoke.
        performance_config_latency: Performance setting (`standard` or `optimized`).
        service_tier: Processing tier (`priority`, `default`, `flex`, or `reserved`).

    For details on the optional parameters, see:
    <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html>

    Returns:
        A dictionary containing the model response.

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
    """
    body = _apply_direct_conversions(body, model_id)
    body = _apply_recursive_conversions(body, model_id)

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
    return json.loads(response['body'].read())


@pxt.udf(is_deterministic=False)
async def invoke_model_async(
    body: dict, *, model_id: str, output_location: str, poll_interval_secs: float = 5.0
) -> dict:
    """
    Invoke a Bedrock model asynchronously via the StartAsyncInvoke API.

    Unlike `invoke_model`, which is a single synchronous HTTP round-trip, `invoke_model_async`
    submits a job and returns immediately with an ARN. Bedrock processes the request in the
    background and writes the result directly to S3 — there is no response body on the HTTP
    connection. This UDF polls until the job completes, reads the result from S3, deletes the
    staging S3 object (since the result is stored in the Pixeltable computed column), and
    returns the result as a dict.

    Use this for models that have no synchronous path:
    - `amazon.nova-reel-v1:0/1` — video generation (output is mp4 in S3)
    - `luma.ray-v2:0` — video generation (output is mp4 in S3)
    - `twelvelabs.marengo-embed-3-0-v1:0` — audio and video embeddings

    Also useful for `amazon.nova-2-multimodal-embeddings-v1:0` with audio/video inputs > 25 MB,
    which exceed the synchronous InvokeModel payload limit.

    PIL images and file paths in the request body are converted to base64 automatically using
    the same model-aware logic as `invoke_model`.

    __Requirements:__

    - `pip install boto3`
    - The AWS credentials used must have `s3:PutObject`, `s3:GetObject`, and `s3:DeleteObject`
      permissions on the `output_location` bucket.

    Args:
        body: The prompt and inference parameters as a dictionary.
        model_id: The model identifier to invoke.
        output_location: S3 URI where Bedrock writes the result, e.g. `s3://my-bucket/prefix`.
            Bedrock requires this — unlike `invoke_model`, the result is never returned inline.
            Each invocation writes to a unique sub-path under this prefix. The staging object
            is deleted after the result is read and stored in the computed column.
        poll_interval_secs: Seconds between `GetAsyncInvoke` status checks. Defaults to 5.0.

    Returns:
        A dictionary containing the model response, same structure as `invoke_model`.

    Examples:
        TwelveLabs Marengo — audio embedding:

        >>> body = {
        ...     'inputType': 'audio',
        ...     'audio': {'mediaSource': {'base64String': t.audio}},
        ... }
        >>> t.add_computed_column(
        ...     response=invoke_model_async(
        ...         body,
        ...         model_id='twelvelabs.marengo-embed-3-0-v1:0',
        ...         output_location='s3://my-bucket/bedrock-output',
        ...     )
        ... )

        TwelveLabs Marengo — video embedding:

        >>> body = {
        ...     'inputType': 'video',
        ...     'video': {'mediaSource': {'base64String': t.video}},
        ... }
        >>> t.add_computed_column(
        ...     response=invoke_model_async(
        ...         body,
        ...         model_id='twelvelabs.marengo-embed-3-0-v1:0',
        ...         output_location='s3://my-bucket/bedrock-output',
        ...     )
        ... )

        Amazon Nova Reel — text-to-video generation:

        >>> body = {
        ...     'taskType': 'TEXT_VIDEO',
        ...     'textToVideoParams': {'text': t.prompt},
        ...     'videoGenerationConfig': {'durationSeconds': 6, 'fps': 24},
        ... }
        >>> t.add_computed_column(
        ...     response=invoke_model_async(
        ...         body,
        ...         model_id='amazon.nova-reel-v1:1',
        ...         output_location='s3://my-bucket/bedrock-output',
        ...     )
        ... )
    """
    body = _apply_direct_conversions(body, model_id)
    body = _apply_recursive_conversions(body, model_id)

    # Submit the async job. Bedrock writes the result to s3_output_uri when done.
    response = await asyncio.to_thread(
        _bedrock_client().start_async_invoke,
        modelId=model_id,
        modelInput=body,
        outputDataConfig={'s3OutputDataConfig': {'s3Uri': output_location}},
    )
    invocation_arn: str = response['invocationArn']
    invocation_id: str = invocation_arn.rsplit('/', maxsplit=1)[-1]

    # Poll until terminal state.
    while True:
        await asyncio.sleep(poll_interval_secs)
        job: dict[str, Any] = await asyncio.to_thread(_bedrock_client().get_async_invoke, invocationArn=invocation_arn)
        status: str = job['status']  # 'InProgress' | 'Completed' | 'Failed'
        if status == 'Completed':
            break
        if status == 'Failed':
            raise pxt.Error(f'Async invocation {invocation_id} failed: {job.get("failureMessage", "unknown error")}')

    soa = ObjectPath.parse_object_storage_addr(output_location, allow_obj_name=False)
    result_prefix = f'{soa.prefix}{invocation_id}/'
    store = ObjectOps.get_store(soa, allow_obj_name=False)
    assert isinstance(store, S3Store), f'Expected S3Store for output_location, got {type(store).__name__}'

    listed = await asyncio.to_thread(
        store.client().list_objects_v2,
        Bucket=store.bucket_name,
        Prefix=result_prefix,
    )
    keys = [obj['Key'] for obj in listed.get('Contents', [])]
    if not keys:
        raise pxt.Error(f'No output found at {output_location}/{invocation_id} after job {invocation_id} completed')

    # Skip manifest.json and read the actual output file.
    result_key = next((k for k in keys if not k.endswith('manifest.json')), None)
    if result_key is None:
        raise pxt.Error(f'No output file (only manifest) found for job {invocation_id}')

    obj = await asyncio.to_thread(
        store.client().get_object, Bucket=store.bucket_name, Key=result_key
    )

    if result_key.endswith('.json'):
        raw_text = obj['Body'].read().decode('utf-8')
        result = json.loads(raw_text)
    else:
        # Binary output (e.g. .mp4) — encode as base64
        result = {'output': b64encode(obj['Body'].read()).decode('utf-8')}

    for key in keys:
        await asyncio.to_thread(store.client().delete_object, Bucket=store.bucket_name, Key=key)
    return result


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
    Generate a conversation response using the Bedrock Converse API.

    Equivalent to the AWS Bedrock `converse` API endpoint.
    For additional details, see:
    <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html>

    Supports all models that have Converse API support. See:
    <https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html>

    PIL images and file paths in `messages[*].content[*].image|video|audio.source.bytes`
    are converted to base64 automatically.

    __Requirements:__

    - `pip install boto3`

    Args:
        messages: Input messages in the Converse API format.
        model_id: The model that will complete your prompt.
        system: An optional system prompt (list of text blocks).
        inference_config: Base inference parameters (e.g. `temperature`, `maxTokens`).
        additional_model_request_fields: Additional model-specific inference parameters.
        tool_config: An optional list of Pixeltable tools to use.

    For details on the optional parameters, see:
    <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column using Claude Haiku:

        >>> msgs = [{'role': 'user', 'content': [{'text': tbl.prompt}]}]
        >>> tbl.add_computed_column(
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
    messages = _apply_converse_conversions(messages)

    kwargs: dict[str, Any] = {'messages': messages, 'modelId': model_id}
    if system is not None:
        kwargs['system'] = system
    if inference_config is not None:
        kwargs['inferenceConfig'] = inference_config
    if additional_model_request_fields is not None:
        kwargs['additionalModelRequestFields'] = additional_model_request_fields
    if tool_config is not None:
        kwargs['toolConfig'] = {
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
        Embedding vector as a float32 array.

    Examples:
        Add an embedding index using Amazon Titan:

        >>> tbl.add_embedding_index(
        ...     tbl.text,
        ...     string_embed=embed.using(
        ...         model_id='amazon.titan-embed-text-v2:0', dimensions=512
        ...     ),
        ... )

        Add an embedding index using Cohere:

        >>> tbl.add_embedding_index(
        ...     tbl.text, string_embed=embed.using(model_id='cohere.embed-english-v3')
        ... )
    """
    from botocore.exceptions import ClientError

    body: dict[str, Any]
    if model_id.startswith('amazon.nova'):
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
        if model_id.startswith('amazon.nova'):
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
    if model_id.startswith('amazon.nova'):
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
        if model_id.startswith('amazon.nova'):
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
    """Converts a Bedrock Converse response dict to Pixeltable tool invocation format and calls `tools._invoke()`."""
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
    return pxt_tool_calls if pxt_tool_calls else None


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
