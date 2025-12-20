"""
Pixeltable UDFs for AWS Bedrock AI models.

Provides integration with AWS Bedrock for accessing various foundation models
including Anthropic Claude, Amazon Titan, and other providers.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import pixeltable as pxt
from pixeltable import env, exprs, type_system as ts
from pixeltable.func import Batch, Tools
from pixeltable.utils.code import local_public_names

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


# Default embedding dimensions for Titan models
_titan_embedding_dimensions: dict[str, int] = {
    'amazon.titan-embed-text-v1': 1536,
    'amazon.titan-embed-text-v2:0': 1024,
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
        ... tbl.add_computed_column(response=invoke_model(body, model_id='amazon.titan-embed-text-v2:0'))
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

    # Read and parse the streaming body
    response_body = json.loads(response['body'].read())
    return response_body


@pxt.udf(batch_size=32)
async def embed_titan(
    text: Batch[str],
    *,
    model_id: str,
    dimensions: int | None = None,
    normalize: bool = True,
    embedding_types: list[str] | None = None,
) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Generate embeddings using Amazon Titan embedding models.

    Calls the AWS Bedrock `invoke_model` API for Titan embedding models.
    For additional details, see:
    <https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html>

    __Requirements:__

    - `pip install boto3`

    Args:
        text: Input text to embed.
        model_id: The Titan embedding model identifier (e.g., 'amazon.titan-embed-text-v2:0').
        dimensions: Output embedding dimensions (V2 only). Valid values: 256, 512, 1024.
        normalize: Whether to normalize the embedding (V2 only). Defaults to True.
        embedding_types: List of embedding types to return (V2 only). Valid values: 'float', 'binary'.

    Returns:
        Array of embedding vectors.

    Examples:
        Add a computed column with Titan embeddings:

        >>> tbl.add_computed_column(embed=embed_titan(tbl.text, model_id='amazon.titan-embed-text-v2:0'))

        With custom dimensions:

        >>> tbl.add_computed_column(embed=embed_titan(tbl.text, model_id='amazon.titan-embed-text-v2:0', dimensions=512))
    """
    import json

    client = _bedrock_client()
    is_v2 = 'v2' in model_id

    results = []
    for t in text:
        # Build request body based on model version
        body: dict[str, Any] = {'inputText': t}

        if is_v2:
            if dimensions is not None:
                body['dimensions'] = dimensions
            body['normalize'] = normalize
            if embedding_types is not None:
                body['embeddingTypes'] = embedding_types

        response = await asyncio.to_thread(
            client.invoke_model,
            body=json.dumps(body),
            modelId=model_id,
            contentType='application/json',
            accept='application/json',
        )

        response_body = json.loads(response['body'].read())
        embedding = response_body['embedding']
        results.append(np.array(embedding, dtype=np.float64))

    return results


@embed_titan.conditional_return_type
def _(
    model_id: str,
    dimensions: int | None = None,
    normalize: bool = True,
    embedding_types: list[str] | None = None,
) -> ts.ArrayType:
    if dimensions is not None:
        return ts.ArrayType((dimensions,), dtype=ts.FloatType(), nullable=False)
    if model_id in _titan_embedding_dimensions:
        return ts.ArrayType((_titan_embedding_dimensions[model_id],), dtype=ts.FloatType(), nullable=False)
    return ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False)


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
