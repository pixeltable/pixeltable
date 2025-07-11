import io
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

from pixeltable import exceptions as excs
from pixeltable.utils.media_store import TempStore


class S3ClientContainer:
    """Contains a group of clients for a service
    New clients are created lazily when needed
    Client access is thread-safe
    Clients can be used for any purpose, but for performance reasons
    they are separated into two groups, one for read and one for write.
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

    def __init__(self, max_read_clients: int = 1, max_write_clients: int = 2, max_pool_connections: int = 5) -> None:
        self.client_config = {
            'max_pool_connections': max_pool_connections,
            'connect_timeout': 15,
            'read_timeout': 30,
            'retries': {'max_attempts': 3, 'mode': 'adaptive'},
        }
        self.client_read = self.ProtectedClientPool(max_read_clients, self.client_config)
        self.client_write = self.ProtectedClientPool(max_write_clients, self.client_config)

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
        return boto3.resource('s3')

    @classmethod
    def parse_uri(cls, source_uri: str) -> tuple[str, str, str]:
        """Parse a URI and return parts
        Returns: scheme, bucket_name, prefix"""
        parsed = urllib.parse.urlparse(source_uri)
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip('/')
        return parsed.scheme, bucket_name, prefix

    def list_objects(self, source_uri: str, n_max: int = 10) -> list[str]:
        """Return a list of objects found with the specified S3 bucket/prefix
        Each returned object includes the full set of prefixes."""
        scheme, bucket_name, prefix = self.parse_uri(source_uri)
        # I think the n_max parameter should be passed into the list_objects_v2 call
        assert scheme == 's3'
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
                    r.append(f'{obj["Key"]}')
        except ClientError as e:
            self.handle_s3_error(e, bucket_name, f'list objects from {source_uri}')
        return r

    def list_uris(self, source_uri: str, n_max: int = 10) -> list[str]:
        """Return a list of URIs found within the specified S3 bucket/path"""
        scheme, bucket_name, _ = self.parse_uri(source_uri)
        assert scheme == 's3'
        objects = self.list_objects(source_uri, n_max)
        r = [f'{scheme}://{bucket_name}/{obj}' for obj in objects]
        return r

    def fetch_url(self, url: str) -> tuple[Optional[str], Optional[Exception]]:
        """Fetches a remote URL into Env.tmp_dir and returns its path"""
        parsed = urllib.parse.urlparse(url)
        # Use len(parsed.scheme) > 1 here to ensure we're not being passed
        # a Windows filename
        assert len(parsed.scheme) > 1 and parsed.scheme != 'file'
        # preserve the file extension, if there is one
        extension = ''
        if parsed.path:
            p = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed.path)))
            extension = p.suffix
        tmp_path = TempStore.create_path(extension=extension)
        try:
            if parsed.scheme == 's3':
                boto_client = self.get_client(for_write=False)
                boto_client.download_file(parsed.netloc, parsed.path.lstrip('/'), str(tmp_path))
            elif parsed.scheme in ('http', 'https'):
                with urllib.request.urlopen(url) as resp, open(tmp_path, 'wb') as f:
                    data = resp.read()
                    f.write(data)
            else:
                raise AssertionError(f'Unsupported URL scheme: {parsed.scheme}')
            return str(tmp_path), None
        except Exception as e:
            # we want to add the file url to the exception message
            exc = excs.Error(f'Failed to download {url}: {e}')
            return None, exc

    def copy_to_s3(self, local_path: Path, remote_uri: str) -> None:
        """Copy a local file to S3"""
        parsed_location = urllib.parse.urlparse(remote_uri)
        if parsed_location.scheme != 's3':
            raise excs.Error(f'Unsupported destination: {remote_uri}')

        bucket = parsed_location.netloc
        remote_dir = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed_location.path)))
        remote_path = str(remote_dir / local_path.name)[1:]  # Remove initial / if it exists

        s3_client = self.get_client(for_write=True)
        upload_args = {'ChecksumAlgorithm': 'SHA256'}
        s3_client.upload_file(Filename=str(local_path), Bucket=bucket, Key=remote_path, ExtraArgs=upload_args)

    def write_to_remote(self, data: io.BytesIO, remote_uri: str, object_name: str) -> str:
        """Send data to remote storage (e.g., S3), and return the URI of the data"""
        scheme, bucket_name, prefix = self.parse_uri(remote_uri)  # Ensure the URI is valid
        if scheme != 's3':
            raise excs.Error(f'Unsupported destination: {remote_uri}')

        remote_dir = Path(urllib.parse.unquote(urllib.request.url2pathname(prefix)))
        remote_path = str(remote_dir / object_name)  # Remove initial / if it exists

        s3_client = self.get_client(for_write=True)
        s3_client.upload_fileobj(data, Bucket=bucket_name, Key=remote_path)
        return f'{scheme}://{bucket_name}/{remote_path}'

    def delete_all_objects_with_prefix(self, bucket_name: str, prefix: str) -> None:
        """
        Delete all objects in the bucket with the given prefix (in batches)
        Uses handle_s3_error for error handling.
        """
        while True:
            target_objects = self.list_objects(f's3://{bucket_name}/{prefix}', n_max=1000)
            if not target_objects or len(target_objects) == 0:
                break
            del_req = {'Objects': [{'Key': obj} for obj in target_objects]}
            try:
                self.get_client(for_write=True).delete_objects(Bucket=bucket_name, Delete=del_req)
            except ClientError as e:
                self.handle_s3_error(e, bucket_name, f'delete objects batch {target_objects}')

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
