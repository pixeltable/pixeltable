from typing import Any


def get_client() -> Any:
    import boto3
    import botocore
    try:
        boto3.Session().get_credentials().get_frozen_credentials()
        return boto3.client('s3')  # credentials are available
    except AttributeError:
        # No credentials available, use unsigned mode
        config = botocore.config.Config(signature_version=botocore.UNSIGNED)
        return boto3.client('s3', config=config)