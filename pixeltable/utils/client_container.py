from __future__ import annotations

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

    def get_client(self, storage_target: str, soa: Optional[StorageObjectAddress]) -> Any:
        """Get the current client for the given target, creating one if needed"""
        with self.client_lock:
            if storage_target not in self.clients:
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
        from .gcs_store import GCSStore
        from .s3_store import S3Store

        if storage_target == 'r2':
            assert soa is not None
            return S3Store.create_r2_client(soa)
        if storage_target == 's3':
            return S3Store.create_s3_client()
        if storage_target == 'gs':
            return GCSStore.create_client()
        raise ValueError(f'Unsupported storage target: {storage_target}')

    def create_resource(self, storage_target: str, soa: StorageObjectAddress) -> Any:
        from .s3_store import S3Store

        if storage_target == 'r2':
            client_args = S3Store.get_r2_client_args(soa)
            return S3Store.create_boto_resource(client_args)
        if storage_target == 's3':
            return S3Store.create_boto_resource({})
        raise ValueError(f'Unsupported storage target: {storage_target}')
