from __future__ import annotations

import threading
import urllib.parse
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from botocore.exceptions import ClientError

from pixeltable import exceptions as excs
from pixeltable.env import Env


class S3ClientContainer:
    """Singleton providing clients and "s3resources" for the S3 storage service
    Client acquisition via the get_client(for_write: bool) method is thread-safe
    S3 resource acquisition via the get_resource() method is not thread-safe

    Implementation:
        Clients are stored in two pools, separated by purpose (read / write)
        The are created as requested, and kept throughout the life of the singleton
        There is only one S3 resource, which is created on first request and reused

    Clients can be used for any purpose, but it is recommended to use different clients for read and write.
    """

    class ProtectedClientPool:
        """An array of clients which can be used to access the S3 resource (thread-safe)"""

        client: list[Optional[Any]]
        client_lock: threading.Lock  # Protects creation of the read client
        round_robin: int  # Used to select a slot in the pool
        client_config: dict[str, Any]  # Used when creating a new client

        def __init__(self, max_clients: int = 1, client_config: Optional[dict[str, Any]] = None) -> None:
            """This init method is not thread safe"""
            self.client = [None for _ in range(max_clients)]
            self.client_lock = threading.Lock()
            self.round_robin = 0
            self.client_config = client_config if client_config is not None else {}

        def get_client(self) -> Any:
            """Get a client, creating one if needed"""
            # This addition below may be an unprotected increment of a state variable across threads
            # So the result is unpredictable, though it will be an integer
            # While written as round-robin, it may also be considered to be a random selection.
            # Either way works.
            # This non-issue would be different if we choose to require and use the 'atomics' module
            self.round_robin += 1
            index = self.round_robin % len(self.client)
            if self.client[index] is not None:
                return self.client[index]

            with self.client_lock:
                if self.client[index] is not None:
                    return self.client[index]
                client = S3ClientContainer.get_client_raw(**self.client_config)

                # Do not set the visible client until it is completely created
                self.client[index] = client
            return self.client[index]

    client_read: ProtectedClientPool
    client_write: ProtectedClientPool
    client_config: dict[str, Any]
    resource: Optional[Any]
    __instance: Optional[S3ClientContainer] = None

    @classmethod
    def get(cls) -> S3ClientContainer:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self, max_clients: Optional[int] = None, max_pool_connections: Optional[int] = None) -> None:
        cpu_count = Env.get().cpu_count
        if max_clients is None:
            max_clients = min(3, cpu_count)
        if max_pool_connections is None:
            total_connections = max(5, 2 * cpu_count)
        else:
            total_connections = max_pool_connections * max_clients
        max_pool_connections = total_connections // max_clients
        self.client_config = {
            'max_pool_connections': max_pool_connections,
            'connect_timeout': 15,
            'read_timeout': 30,
            'retries': {'max_attempts': 3, 'mode': 'adaptive'},
        }
        self.client_read = self.ProtectedClientPool(max_clients, self.client_config)
        self.client_write = self.ProtectedClientPool(max_clients, self.client_config)
        self.resource = None

    @classmethod
    def get_client_raw(cls, **kwargs: Any) -> Any:
        """Get a raw client without any locking"""
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

    def get_client(self, for_write: bool) -> Any:
        """Get a client, creating one if needed"""
        if for_write:
            return self.client_write.get_client()
        else:
            return self.client_read.get_client()

    def get_resource(self) -> Any:
        """Return the current S3 resource, creating it if needed"""
        if self.resource is None:
            import boto3

            self.resource = boto3.resource('s3')
        return self.resource

    @classmethod
    def parse_uri(cls, source_uri: str) -> tuple[str, str, str]:
        """Parse a URI and return scheme, bucket_name, prefix"""
        parsed = urllib.parse.urlparse(source_uri)
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip('/')
        return parsed.scheme, bucket_name, prefix

    def list_objects(self, source_uri: str, return_uri: bool, n_max: int = 10) -> list[str]:
        """Return a list of objects found with the specified S3 uri
        Each returned object includes the full set of prefixes.
        if return_uri is True, the full S3 URI is returned; otherwise, just the object key.
        """
        scheme, bucket_name, prefix = self.parse_uri(source_uri)
        # I think the n_max parameter should be passed into the list_objects_v2 call
        assert scheme == 's3'
        p = f'{scheme}://{bucket_name}/' if return_uri else ''
        s3_client = self.get_client(for_write=False)
        r: list[str] = []
        try:
            # Use paginator to handle more than 1000 objects
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                for obj in page['Contents']:
                    if len(r) >= n_max:
                        return r
                    r.append(f'{p}{obj["Key"]}')
        except ClientError as e:
            self.handle_s3_error(e, bucket_name, f'list objects from {source_uri}')
        return r

    def list_uris(self, source_uri: str, n_max: int = 10) -> list[str]:
        """Return a list of URIs found within the specified S3 uri"""
        return self.list_objects(source_uri, True, n_max)

    @classmethod
    def handle_s3_error(
        cls, e: ClientError, bucket_name: str, operation: str = '', *, ignore_404: bool = False
    ) -> None:
        error_code = e.response.get('Error', {}).get('Code')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        if ignore_404 and error_code == '404':
            return
        if error_code == '404':
            raise excs.Error(f'Bucket {bucket_name} not found during {operation}: {error_message}')
        elif error_code == '403':
            raise excs.Error(f'Access denied to bucket {bucket_name} during {operation}: {error_message}')
        elif error_code == 'PreconditionFailed' or 'PreconditionFailed' in error_message:
            raise excs.Error(f'Precondition failed for bucket {bucket_name} during {operation}: {error_message}')
        else:
            raise excs.Error(f'Error during {operation} in bucket {bucket_name}: {error_code} - {error_message}')
