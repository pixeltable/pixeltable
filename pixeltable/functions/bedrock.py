"""
Pixeltable UDFs for AWS Bedrock AI models.

Provides integration with AWS Bedrock for accessing various foundation models
including Anthropic Claude, Amazon Titan, and other providers.
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import PIL.Image

import pixeltable as pxt
from pixeltable import env, exprs, type_system as ts
from pixeltable.func import Tools
from pixeltable.utils.code import local_public_names
from pixeltable.utils.image import to_base64

if TYPE_CHECKING:
    from botocore.client import BaseClient

_logger = logging.getLogger('pixeltable')


@env.register_client('bedrock')
def _() -> 'BaseClient':
    import boto3

    return boto3.client(service_name='bedrock-runtime')


# boto3 typing is weird; type information is dynamically defined, so the best we can do for the static checker is `Any`
def _bedrock_client() -> Any:
    return env.Env.get().get_client('bedrock')


# Default embedding dimensions for models
_embedding_dimensions: dict[str, int] = {
    'amazon.titan-embed-text-v1': 1536,
    'amazon.titan-embed-text-v2:0': 1024,
    'amazon.titan-embed-image-v1': 1024,
    'amazon.nova-2-multimodal-embeddings-v1:0': 3072,
}


@pxt.udf
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

    __Requirements:__

    - `pip install boto3`

    Args:
        messages: Input messages.
        model_id: The model that will complete your prompt.
        system: An optional system prompt.
        inference_config: Base inference parameters to use.
        additional_model_request_fields: Additional inference parameters to use.

    For details on the optional parameters, see:
    <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `anthropic.claude-3-haiku-20240307-v1:0`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> msgs = [{'role': 'user', 'content': [{'text': tbl.prompt}]}]
        ... tbl.add_computed_column(response=messages(msgs, model_id='anthropic.claude-3-haiku-20240307-v1:0'))
    """

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

    __Requirements:__

    - `pip install boto3`

    Args:
        body: The prompt and inference parameters as a dictionary.
        model_id: The model identifier to invoke.
        performance_config_latency: Performance setting.
        service_tier: processing tier.

    For details on the optional parameters, see:
    <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html>

    Returns:
        A dictionary containing the model response.

    Examples:
        Add a computed column using Amazon Titan embedding model:

        >>> body = {'inputText': tbl.text, 'dimensions': 512, 'normalize': True}
        >>> tbl.add_computed_column(response=invoke_model(body, model_id='amazon.titan-embed-text-v2:0'))
    """
    import json

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
    response_body = json.loads(response['body'].read())
    return response_body


@pxt.udf
async def embed(text: str, *, model_id: str, dimensions: int | None = None) -> pxt.Array[(None,), np.float32]:
    """
    Generate embeddings using Amazon Titan or Nova embedding models.

    Calls the AWS Bedrock `invoke_model` API for embedding models.
    For additional details, see:
    <https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html>
    <https://docs.aws.amazon.com/nova/latest/userguide/modality-embedding.html>

    __Requirements:__

    - `pip install boto3`

    Args:
        text: Input text to embed.
        model_id: The embedding model identifier.
        dimensions: Output embedding dimensions. For Titan V2: 256, 512, 1024. For Nova: 256, 512, 1024, 3072.

    Returns:
        Embedding vector

    Examples:
        Create an embedding index on a column `description` with Nova embeddings and custom dimensions:

        >>> tbl.add_embedding_index(
        ...     tbl.description,
        ...     string_embed=embed.using(model_id='amazon.nova-2-multimodal-embeddings-v1:0', dimensions=1024)
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
        else:
            return np.array(response_body['embedding'], dtype=np.float32)
    except ClientError as e:
        raise pxt.Error(f'Failed to generate embedding: {e}') from e


@embed.overload
async def _(image: PIL.Image.Image, *, model_id: str, dimensions: int | None = None) -> pxt.Array[(None,), np.float32]:
    from botocore.exceptions import ClientError

    if 'nova' in model_id:
        body: dict[str, Any] = {
            'taskType': 'SINGLE_EMBEDDING',
            'singleEmbeddingParams': {
                'embeddingPurpose': 'GENERIC_INDEX',
                'image': {'format': 'jpeg', 'source': {'bytes': to_base64(image, format='jpeg')}},
            },
        }
        if dimensions is not None:
            body['singleEmbeddingParams']['embeddingDimension'] = dimensions
    else:
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
    """Converts an Anthropic response dict to Pixeltable tool invocation format and calls `tools._invoke()`."""
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
