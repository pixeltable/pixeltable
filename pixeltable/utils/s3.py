from typing import Any


def get_client(**kwargs: Any) -> Any:
    import boto3
    import botocore

    try:
        boto3.Session().get_credentials().get_frozen_credentials()
        config = botocore.config.Config(**kwargs)
        return boto3.client('s3', config=config)  # credentials are available
    except AttributeError:
        # No credentials available, use unsigned mode
        config_args = kwargs.copy()
        config_args['signature_version'] = botocore.UNSIGNED
        config = botocore.config.Config(**config_args)
        return boto3.client('s3', config=config)
