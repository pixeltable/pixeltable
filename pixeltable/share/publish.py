import dataclasses
import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Literal

import requests
from requests.adapters import HTTPAdapter
from rich.progress import BarColumn, DownloadColumn, Progress, TaskID, TransferSpeedColumn
from urllib3.util.retry import Retry

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.catalog import Catalog
from pixeltable.catalog.table_version import TableVersionMd
from pixeltable.env import Env
from pixeltable.utils import sha256sum
from pixeltable.utils.local_store import TempStore

from .packager import TablePackager, TableRestorer
from .protocol import PxtUri
from .protocol.replica import (
    DeleteRequest,
    DeleteResponse,
    FinalizeRequest,
    FinalizeResponse,
    PublishRequest,
    PublishResponse,
    ReplicateRequest,
    ReplicateResponse,
)

_logger = logging.getLogger('pixeltable')


class _ProgressFileReader:
    """File wrapper that tracks read progress for HTTP uploads."""

    def __init__(self, file_obj: BinaryIO, progress: Progress, task_id: TaskID) -> None:
        self.file_obj = file_obj
        self.progress = progress
        self.task_id = task_id

    def read(self, size: int = -1) -> bytes:
        data = self.file_obj.read(size)
        if data:
            self.progress.update(self.task_id, advance=len(data))
        return data

    def __enter__(self) -> '_ProgressFileReader':
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self.file_obj, name)


# These URLs are abstracted out for now, but will be replaced with actual (hard-coded) URLs once the
# pixeltable.com URLs are available.

PIXELTABLE_API_URL = os.environ.get('PIXELTABLE_API_URL', 'https://internal-api.pixeltable.com')


def push_replica(
    dest_tbl_uri: str, src_tbl: pxt.Table, bucket: str | None = None, access: Literal['public', 'private'] = 'private'
) -> str:
    _logger.info(f'Publishing replica for {src_tbl._name!r} to: {dest_tbl_uri}')

    packager = TablePackager(src_tbl)
    # Create the publish request using packager's bundle_md
    publish_request = PublishRequest(
        table_uri=PxtUri(uri=dest_tbl_uri),
        pxt_version=packager.bundle_md['pxt_version'],
        pxt_md_version=packager.bundle_md['pxt_md_version'],
        md=[TableVersionMd.from_dict(md_dict) for md_dict in packager.bundle_md['md']],
        bucket_name=bucket,
        is_public=access == 'public',
    )

    _logger.debug(f'Sending PublishRequest: {publish_request}')

    response = requests.post(PIXELTABLE_API_URL, data=publish_request.model_dump_json(), headers=_api_headers())
    if response.status_code == 201:
        publish_response = PublishResponse.model_validate(response.json())
        existing_table_uri = str(publish_response.table_uri)
        Env.get().console_logger.info(
            f'Replica for version {publish_request.md[0].version_md.version} already exists at {existing_table_uri}.'
        )
        with Catalog.get().begin_xact(tbl_id=src_tbl._id, for_write=True):
            Catalog.get().update_additional_md(src_tbl._id, {'pxt_uri': existing_table_uri})
        return existing_table_uri
    if response.status_code != 200:
        raise excs.Error(f'Error publishing {src_tbl._display_name()}: {response.text}')
    publish_response = PublishResponse.model_validate(response.json())

    _logger.debug(f'Received PublishResponse: {publish_response}')

    upload_id = publish_response.upload_id
    destination_uri = publish_response.destination_uri

    Env.get().console_logger.info(f"Creating a replica of '{src_tbl._path()}' at: {dest_tbl_uri}")

    bundle = packager.package()

    parsed_location = urllib.parse.urlparse(str(destination_uri))
    if parsed_location.scheme == 's3':
        _upload_bundle_to_s3(bundle, parsed_location)
    elif parsed_location.scheme == 'https':
        _upload_to_presigned_url(file_path=bundle, url=parsed_location.geturl())
    else:
        raise excs.Error(f'Unsupported destination: {destination_uri}')

    Env.get().console_logger.info('Finalizing replica ...')
    # Use preview data from packager's bundle_md (set during package())
    finalize_request = FinalizeRequest(
        table_uri=PxtUri(uri=dest_tbl_uri),
        upload_id=upload_id,
        datafile=bundle.name,
        size=bundle.stat().st_size,
        sha256=sha256sum(bundle),  # Generate our own SHA for independent verification
        row_count=packager.bundle_md['row_count'],
        preview_header=packager.bundle_md['preview_header'],
        preview_data=packager.bundle_md['preview_data'],
    )
    finalize_response_json = requests.post(
        PIXELTABLE_API_URL, data=finalize_request.model_dump_json(), headers=_api_headers()
    )
    if finalize_response_json.status_code != 200:
        raise excs.Error(f'Error finalizing {src_tbl._display_name()}: {finalize_response_json.text}')

    finalize_response = FinalizeResponse.model_validate(finalize_response_json.json())
    confirmed_tbl_uri = finalize_response.confirmed_table_uri
    Env.get().console_logger.info(f'The published table is now available at: {confirmed_tbl_uri}')

    with Catalog.get().begin_xact(tbl_id=src_tbl._id, for_write=True):
        Catalog.get().update_additional_md(src_tbl._id, {'pxt_uri': str(confirmed_tbl_uri)})

    return str(confirmed_tbl_uri)


def _upload_bundle_to_s3(bundle: Path, parsed_location: urllib.parse.ParseResult) -> None:
    bucket = parsed_location.netloc
    remote_dir = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed_location.path)))
    remote_path = str(remote_dir / bundle.name)[1:]  # Remove initial /

    Env.get().console_logger.info(f'Uploading replica to: {bucket}:{remote_path}')

    s3_client = Env.get().get_client('s3')

    upload_args = {'ChecksumAlgorithm': 'SHA256'}

    with Progress(BarColumn(), DownloadColumn(), TransferSpeedColumn()) as progress:
        task_id = progress.add_task('Uploading', total=bundle.stat().st_size)
        s3_client.upload_file(
            Filename=str(bundle),
            Bucket=bucket,
            Key=remote_path,
            ExtraArgs=upload_args,
            Callback=lambda n: progress.update(task_id, advance=n),
        )


def pull_replica(dest_path: str, src_tbl_uri: str) -> pxt.Table:
    parsed_uri = PxtUri(src_tbl_uri)
    clone_request = ReplicateRequest(table_uri=parsed_uri)
    response = requests.post(PIXELTABLE_API_URL, data=clone_request.model_dump_json(), headers=_api_headers())
    if response.status_code != 200:
        raise excs.Error(f'Error cloning replica: {response.text}')
    clone_response = ReplicateResponse.model_validate(response.json())

    # Prevalidate destination path for replication. We do this before downloading the bundle so that we avoid
    # having to download it if there is a collision or if this is a duplicate replica. This is done outside the
    # transaction scope of the table restore operation (we don't want to hold a transaction open during the
    # download); that's fine, since it will be validated again during TableRestorer's catalog operations.

    t = pxt.get_table(dest_path, if_not_exists='ignore')
    if t is not None:
        if str(t._id) != clone_response.md[0].tbl_md.tbl_id:
            raise excs.Error(
                f'An attempt was made to create a replica table at {dest_path!r}, '
                'but a different table already exists at that location.'
            )
        known_versions = tuple(v['version'] for v in t.get_versions())
        if clone_response.md[0].version_md.version in known_versions:
            Env.get().console_logger.info(f'Replica {dest_path!r} is already up to date with source: {src_tbl_uri}')
            return t

    primary_version_additional_md = clone_response.md[0].version_md.additional_md
    bundle_uri = str(clone_response.destination_uri)
    bundle_filename = primary_version_additional_md['cloud']['datafile']
    parsed_location = urllib.parse.urlparse(bundle_uri)
    if parsed_location.scheme == 's3':
        bundle_path = _download_bundle_from_s3(parsed_location, bundle_filename)
    elif parsed_location.scheme == 'https':
        bundle_path = TempStore.create_path()
        _download_from_presigned_url(url=parsed_location.geturl(), output_path=bundle_path)
    else:
        raise excs.Error(f'Unexpected response from server: unsupported bundle uri: {bundle_uri}')

    pxt_uri = str(clone_response.table_uri)
    md_list = [dataclasses.asdict(md) for md in clone_response.md]
    restorer = TableRestorer(
        dest_path, {'pxt_version': pxt.__version__, 'pxt_md_version': clone_response.pxt_md_version, 'md': md_list}
    )

    tbl = restorer.restore(bundle_path, pxt_uri, explicit_version=parsed_uri.version)
    Env.get().console_logger.info(f'Created local replica {tbl._path()!r} from URI: {src_tbl_uri}')
    return tbl


def _download_bundle_from_s3(parsed_location: urllib.parse.ParseResult, bundle_filename: str) -> Path:
    bucket = parsed_location.netloc
    remote_dir = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed_location.path)))
    remote_path = str(remote_dir / bundle_filename)[1:]  # Remove initial /

    Env.get().console_logger.info(f'Downloading replica from: {bucket}:{remote_path}')

    s3_client = Env.get().get_client('s3')

    obj = s3_client.head_object(Bucket=bucket, Key=remote_path)  # Check if the object exists
    bundle_size = obj['ContentLength']

    bundle_path = TempStore.create_path()
    with Progress(BarColumn(), DownloadColumn(), TransferSpeedColumn()) as progress:
        task_id = progress.add_task('Downloading', total=bundle_size)
        s3_client.download_file(
            Bucket=bucket,
            Key=remote_path,
            Filename=str(bundle_path),
            Callback=lambda n: progress.update(task_id, advance=n),
        )
    return bundle_path


def _create_retry_session(
    max_retries: int = 3, backoff_factor: float = 1.0, status_forcelist: list | None = None
) -> requests.Session:
    """Create a requests session with retry configuration"""
    if status_forcelist is None:
        status_forcelist = [
            408,  # Request Timeout
            429,  # Too Many Requests (rate limiting)
            500,  # Internal Server Error (server-side error)
            502,  # Bad Gateway (proxy/gateway got invalid response)
            503,  # Service Unavailable (server overloaded or down)
            504,  # Gateway Timeout (proxy/gateway timeout)
        ]
    retry_strategy = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=['GET', 'PUT', 'POST', 'DELETE'],
    )

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    return session


def _upload_to_presigned_url(file_path: Path, url: str, max_retries: int = 3) -> requests.Response:
    """Upload file with progress bar and retries"""
    file_size = file_path.stat().st_size

    headers = {'Content-Length': str(file_size), 'Content-Type': 'application/octet-stream'}

    # Detect if it's Azure by URL pattern
    is_azure = 'blob.core.windows.net' in url
    if is_azure:
        headers['x-ms-blob-type'] = 'BlockBlob'

    session = _create_retry_session(max_retries=max_retries)
    try:
        with Progress(BarColumn(), DownloadColumn(), TransferSpeedColumn()) as progress:
            task_id = progress.add_task('Uploading', total=file_size)
            with open(file_path, 'rb') as f:
                file_with_progress = _ProgressFileReader(f, progress, task_id)
                response = session.put(
                    url,
                    data=file_with_progress,
                    headers=headers,
                    timeout=(60, 1800),  # 60 seconds to connect and 1800 seconds for server response
                )
                response.raise_for_status()
                return response
    finally:
        session.close()


def _download_from_presigned_url(
    url: str, output_path: Path, headers: dict[str, str] | None = None, max_retries: int = 3
) -> None:
    """Download file with progress bar and retries"""
    session = _create_retry_session(max_retries=max_retries)

    try:
        # Stream download with progress
        response = session.get(
            url, headers=headers, stream=True, timeout=(60, 300)
        )  # 60 seconds to connect and 300 seconds for server response
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with Progress(BarColumn(), DownloadColumn(), TransferSpeedColumn()) as progress:
            task_id = progress.add_task('Downloading', total=total_size)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))
    finally:
        session.close()


def delete_replica(dest_path: str, version: int | None = None) -> None:
    """Delete cloud replica"""
    delete_request = DeleteRequest(table_uri=PxtUri(uri=dest_path), version=version)
    response = requests.post(PIXELTABLE_API_URL, data=delete_request.model_dump_json(), headers=_api_headers())
    if response.status_code != 200:
        raise excs.Error(f'Error deleting replica: {response.text}')
    DeleteResponse.model_validate(response.json())
    Env.get().console_logger.info(f'Deleted replica at: {dest_path}')


def list_table_versions(table_uri: str) -> list[dict[str, Any]]:
    """List versions for a remote table."""
    request_json = {'operation_type': 'list_table_versions', 'table_uri': {'uri': table_uri}}
    response = requests.post(PIXELTABLE_API_URL, data=json.dumps(request_json), headers=_api_headers())
    if response.status_code != 200:
        raise excs.Error(f'Error listing table versions: {response.text}')
    response_data = response.json()
    return response_data.get('versions', [])


def _api_headers() -> dict[str, str]:
    headers = {'Content-Type': 'application/json'}
    api_key = Env.get().pxt_api_key
    if api_key is not None:
        headers['X-api-key'] = api_key
    return headers
