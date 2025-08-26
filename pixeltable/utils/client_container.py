from __future__ import annotations

import os
import threading
from typing import Any, Optional

from pixeltable.env import Env
from pixeltable.utils.media_path import StorageObjectAddress


class ClientContainer:
    """Singleton providing clients and resources for storage services.
    Methods are thread-safe.
    Client acquisition via the get_client(for_write: bool) method is thread-safe
    S3 resource acquisition via the get_resource() method is not thread-safe

    Implementation:
        Clients are stored in two pools, separated by purpose (read / write)
        The are created as requested, and kept throughout the life of the singleton
        There is only one S3 resource, which is created on first request and reused

    Clients can be used for any purpose, but it is recommended to use different clients for read and write.
    """

    client_lock: threading.Lock  # Protects creation of the clients
    clients: dict[str, Any]  # Maps storage_target to client
    resources: dict[str, Any]  # Maps storage_target to resource
    client_max_connections: int  # Total number of connections when creating client

    client_config: dict[str, Any]
    __instance: Optional[ClientContainer] = None

    @classmethod
    def get(cls) -> ClientContainer:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self) -> None:
        assert ClientContainer.__instance is None
        self.client_lock = threading.Lock()
        self.clients = {}
        self.resources = {}

        cpu_count = Env.get().cpu_count
        self.client_max_connections = max(5, 4 * cpu_count)

    def get_client(self, for_write: bool, storage_target: str, soa: Optional[StorageObjectAddress]) -> Any:
        """Get the current client for the given target, creating one if needed"""
        with self.client_lock:
            if storage_target not in self.clients:
                # Ignore for_write parameter
                self.clients[storage_target] = self.create_client(storage_target, soa)
                assert storage_target in self.clients
            return self.clients[storage_target]

    def get_resource(self, storage_target: str, soa: StorageObjectAddress) -> Any:
        """Return the current resource for the given target, creating one if needed"""
        with self.client_lock:
            if storage_target not in self.resources:
                self.resources[storage_target] = self.create_resource(storage_target, soa)
            return self.resources[storage_target]

    def create_client(self, storage_target: str, soa: Optional[StorageObjectAddress]) -> Any:
        """Create a new client for the given storage target and storage object address."""
        if storage_target == 'r2':
            assert soa is not None
            return self.create_r2_client(soa)
        if storage_target == 's3':
            return self.create_s3_client()
        if storage_target == 'gs':
            return self.create_gs_client()
        raise ValueError(f'Unsupported storage target: {storage_target}')

    def create_s3_client(self) -> Any:
        """Get a raw client without any locking"""
        client_args: dict[str, Any] = {}
        client_config = {
            'max_pool_connections': self.client_max_connections,
            'connect_timeout': 15,
            'read_timeout': 30,
            'retries': {'max_attempts': 3, 'mode': 'adaptive'},
        }
        return self.get_boto_client(client_args, client_config)

    def get_r2_client_args(self, soa: StorageObjectAddress) -> dict[str, Any]:
        client_args = {}
        if soa.storage_target == 'r2':
            a_key = os.getenv('R2_ACCESS_KEY', '')
            s_key = os.getenv('R2_SECRET_KEY', '')
            if a_key and s_key:
                client_args = {
                    'aws_access_key_id': a_key,
                    'aws_secret_access_key': s_key,
                    'region_name': 'auto',
                    'endpoint_url': soa.container_free_uri,
                }
        return client_args

    def create_r2_client(self, soa: StorageObjectAddress) -> Any:
        client_args = self.get_r2_client_args(soa)
        client_config = {
            'max_pool_connections': self.client_max_connections,
            'connect_timeout': 15,
            'read_timeout': 30,
            'retries': {'max_attempts': 3, 'mode': 'adaptive'},
        }
        return self.get_boto_client(client_args, client_config)

    @classmethod
    def get_boto_client(cls, client_args: dict[str, Any], config_args: dict[str, Any]) -> Any:
        import boto3
        import botocore

        try:
            if len(client_args) == 0:
                # No client args supplibed, attempt to get default (s3) credentials
                boto3.Session().get_credentials().get_frozen_credentials()
                config = botocore.config.Config(**config_args)
                return boto3.client('s3', config=config)  # credentials are available
            else:
                # If client args are provided, use them directly
                config = botocore.config.Config(**config_args)
                return boto3.client('s3', **client_args, config=config)
        except AttributeError:
            # No credentials available, use unsigned mode
            config_args = config_args.copy()
            config_args['signature_version'] = botocore.UNSIGNED
            config = botocore.config.Config(**config_args)
            return boto3.client('s3', config=config)

    def create_resource(self, storage_target: str, soa: StorageObjectAddress) -> Any:
        if storage_target == 'r2':
            client_args = self.get_r2_client_args(soa)
            return self.create_boto_resource(client_args)
        if storage_target == 's3':
            return self.create_boto_resource({})
        raise ValueError(f'Unsupported storage target: {storage_target}')

    def create_boto_resource(self, client_args: dict[str, Any]) -> Any:
        import boto3

        return boto3.resource('s3', **client_args)

    @classmethod
    def create_gs_client(cls) -> Any:
        from google.cloud import storage  # type: ignore[attr-defined]

        try:
            # Try to create client with default credentials
            client = storage.Client()
            return client
        except Exception:
            # If no credentials are available, create anonymous client for public buckets
            client = storage.Client.create_anonymous_client()
            return client
