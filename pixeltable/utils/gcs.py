"""gcs.py - Google Cloud Storage client management equivalent to s3.py"""

from __future__ import annotations

import threading
import urllib.parse
from typing import Any, Optional

from pixeltable import exceptions as excs
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

    @classmethod
    def parse_uri(cls, source_uri: str) -> tuple[str, str, str]:
        """Parse a URI and return scheme, bucket_name, prefix"""
        parsed = urllib.parse.urlparse(source_uri)
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip('/')
        return parsed.scheme, bucket_name, prefix

    def list_objects(self, source_uri: str, return_uri: bool, n_max: int = 10) -> list[str]:
        """Return a list of objects found with the specified GCS uri
        Each returned object includes the full set of prefixes.
        if return_uri is True, the full GCS URI is returned; otherwise, just the object key.
        """
        from google.api_core.exceptions import GoogleAPIError

        scheme, bucket_name, prefix = self.parse_uri(source_uri)
        assert scheme == 'gs'
        p = f'{scheme}://{bucket_name}/' if return_uri else ''
        gcs_client = self.get_client(for_write=False)
        r: list[str] = []

        try:
            bucket = gcs_client.bucket(bucket_name)
            # List blobs with the given prefix, limiting to n_max
            blobs = bucket.list_blobs(prefix=prefix, max_results=n_max)

            for blob in blobs:
                r.append(f'{p}{blob.name}')
                if len(r) >= n_max:
                    break

        except GoogleAPIError as e:
            self.handle_gcs_error(e, bucket_name, f'list objects from {source_uri}')
        return r

    def list_uris(self, source_uri: str, n_max: int = 10) -> list[str]:
        """Return a list of URIs found within the specified GCS uri"""
        return self.list_objects(source_uri, True, n_max)

    @classmethod
    def handle_gcs_error(cls, e: Exception, bucket_name: str, operation: str = '', *, ignore_404: bool = False) -> None:
        """Handle GCS-specific errors and convert them to appropriate exceptions"""
        from google.api_core.exceptions import GoogleAPIError
        from google.cloud.exceptions import Forbidden, NotFound

        if isinstance(e, NotFound):
            if ignore_404:
                return
            raise excs.Error(f'Bucket or object {bucket_name} not found during {operation}: {str(e)!r}')
        elif isinstance(e, Forbidden):
            raise excs.Error(f'Access denied to bucket {bucket_name} during {operation}: {str(e)!r}')
        elif isinstance(e, GoogleAPIError):
            # Handle other Google API errors
            error_message = str(e)
            if 'Precondition' in error_message:
                raise excs.Error(f'Precondition failed for bucket {bucket_name} during {operation}: {error_message}')
            else:
                raise excs.Error(f'Error during {operation} in bucket {bucket_name}: {error_message}')
        else:
            # Generic error handling
            raise excs.Error(f'Unexpected error during {operation} in bucket {bucket_name}: {str(e)!r}')
