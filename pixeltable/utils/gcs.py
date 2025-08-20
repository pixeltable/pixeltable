"""gcs.py - Google Cloud Storage client management equivalent to s3.py"""

from __future__ import annotations

import threading
from typing import Any, Optional

from pixeltable.env import Env


class GCSClientContainer:
    """Singleton providing clients for the Google Cloud Storage service
    Client acquisition via the get_client(for_write: bool) method is thread-safe

    Implementation:
        Clients are stored in two pools, separated by purpose (read / write)
        They are created as requested, and kept throughout the life of the singleton

    Clients can be used for any purpose, but it is recommended to use different clients for read and write.
    """

    class ProtectedClientPool:
        """An array of clients which can be used to access the GCS resource (thread-safe)"""

        client: list[Optional[Any]]
        client_lock: threading.Lock  # Protects creation of the client
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
            self.round_robin += 1
            index = self.round_robin % len(self.client)
            if self.client[index] is not None:
                return self.client[index]

            with self.client_lock:
                if self.client[index] is not None:
                    return self.client[index]
                client = GCSClientContainer.get_client_raw(**self.client_config)

                # Do not set the visible client until it is completely created
                self.client[index] = client
            return self.client[index]

    client_read: ProtectedClientPool
    client_write: ProtectedClientPool
    client_config: dict[str, Any]
    __instance: Optional[GCSClientContainer] = None

    @classmethod
    def get(cls) -> GCSClientContainer:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self, max_clients: Optional[int] = None) -> None:
        cpu_count = Env.get().cpu_count
        if max_clients is None:
            max_clients = min(3, cpu_count)

        # GCS client configuration
        self.client_config = {
            'timeout': 30
        }  # , 'retry': Retry(initial=1.0, maximum=60.0, multiplier=2.0, deadline=300.0)}
        self.client_read = self.ProtectedClientPool(max_clients, self.client_config)
        self.client_write = self.ProtectedClientPool(max_clients, self.client_config)

    @classmethod
    def get_client_raw(cls, **kwargs: Any) -> Any:
        """Get a raw client without any locking"""
        from google.cloud import storage  # type: ignore[attr-defined]

        try:
            # Try to create client with default credentials
            client = storage.Client()
            return client
        except Exception:
            # If no credentials are available, create anonymous client for public buckets
            client = storage.Client.create_anonymous_client()
            return client

    def get_client(self, for_write: bool) -> Any:
        """Get a client, creating one if needed"""
        if for_write:
            return self.client_write.get_client()
        else:
            return self.client_read.get_client()
