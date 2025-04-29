import logging
from typing import TYPE_CHECKING, Any, Optional

import pixeltable as pxt
from pixeltable import env, exprs
from pixeltable.func import Tools
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import boto3
    from botocore.client import BaseClient

_logger = logging.getLogger('pixeltable')


@env.register_client('bedrock')
def _() -> 'BaseClient':
    import boto3

    return boto3.client(service_name='bedrock-runtime')


def _bedrock_client() -> 'BaseClient':
    return env.Env.get().get_client('bedrock')


@pxt.udf
def converse(
    messages: list[dict[str, Any]],
    *,
    model_id: str,
    system: Optional[list[dict[str, Any]]] = None,
    inference_config: Optional[dict] = None,
    additional_model_request_fields: Optional[dict] = None,
) -> dict:
    """
    Generate a conversation response.

    Equivalent to the AWS Bedrock `converse` API endpoint.
    For additional details, see: <https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html>

    __Requirements:__

    - `pip install boto3`

    Args:
        messages: Input messages.
        model_id: The model that will complete your prompt.
        system: An optional system prompt.
        inference_config: Base inference parameters to use.
        additional_model_request_fields: Additional inference parameters to use.

    For details on the optional parameters, see:
    <https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `anthropic.claude-3-haiku-20240307-v1:0`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> msgs = [{'role': 'user', 'content': [{'text': tbl.prompt}]}]
        ... tbl.add_computed_column(response=messages(msgs, model_id='anthropic.claude-3-haiku-20240307-v1:0'))
    """

    if system is None:
        system = []
    if inference_config is None:
        inference_config = {}
    if additional_model_request_fields is None:
        additional_model_request_fields = {}

    return _bedrock_client().converse(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_request_fields,
    )
