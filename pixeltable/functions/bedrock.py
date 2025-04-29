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
