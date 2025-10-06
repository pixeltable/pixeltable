import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal, Optional

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.utils import sha256sum
from pixeltable.utils.local_store import TempStore

from .packager import TablePackager, TableRestorer
from .protocol import PxtUri
from .protocol.replica import (
    DeleteRequest,
    FinalizeRequest,
    FinalizeResponse,
    PublishRequest,
    PublishResponse,
    ReplicateRequest,
    ReplicateResponse,
)

# These URLs are abstracted out for now, but will be replaced with actual (hard-coded) URLs once the
# pixeltable.com URLs are available.

PIXELTABLE_API_URL = os.environ.get('PIXELTABLE_API_URL', 'https://internal-api.pixeltable.com')


def push_replica(
    dest_tbl_uri: str, src_tbl: pxt.Table, bucket: str | None = None, access: Literal['public', 'private'] = 'private'
) -> str:
    packager = TablePackager(src_tbl)

    # Create the publish request using packager's bundle_md
    publish_request = PublishRequest(
        table_uri=PxtUri(dest_tbl_uri),
        pxt_version=packager.bundle_md['pxt_version'],
        pxt_md_version=packager.bundle_md['pxt_md_version'],
        md=packager.bundle_md['md'],
        bucket_name=bucket,
        is_public=access == 'public',
    )

    response = requests.post(PIXELTABLE_API_URL, json=publish_request.model_dump_json(), headers=_api_headers())
    if response.status_code != 200:
        raise excs.Error(f'Error publishing snapshot: {response.text}')
    publish_response = PublishResponse.model_validate_json(response.json())

    upload_id = publish_response.upload_id
    destination_uri = publish_response.destination_uri

    Env.get().console_logger.info(f"Creating a snapshot of '{src_tbl._path()}' at: {dest_tbl_uri}")

    bundle = packager.package()

    parsed_location = urllib.parse.urlparse(str(destination_uri))
    if parsed_location.scheme == 's3':
        _upload_bundle_to_s3(bundle, parsed_location)
    elif parsed_location.scheme == 'https':
        _upload_to_presigned_url(file_path=bundle, url=parsed_location.geturl())
    else:
        raise excs.Error(f'Unsupported destination: {destination_uri}')

    Env.get().console_logger.info('Finalizing snapshot ...')
    # Use preview data from packager's bundle_md (set during package())
    finalize_request = FinalizeRequest(
        table_uri=PxtUri(dest_tbl_uri),
        upload_id=upload_id,
        datafile=bundle.name,
        size=bundle.stat().st_size,
        sha256=sha256sum(bundle),  # Generate our own SHA for independent verification
        row_count=packager.bundle_md['row_count'],
        preview_header=packager.bundle_md['preview_header'],
        preview_data=packager.bundle_md['preview_data'],
    )
    finalize_response = requests.post(
        PIXELTABLE_API_URL, json=finalize_request.model_dump_json(), headers=_api_headers()
    )
    if finalize_response.status_code != 200:
        raise excs.Error(f'Error finalizing snapshot: {finalize_response.text}')
    finalize_response = FinalizeResponse.model_validate_json(finalize_response.json())
    Env.get().console_logger.info(f'The published snapshot is now available at:{finalize_response.confirmed_table_uri}')
    return str(finalize_response.confirmed_table_uri)


def _upload_bundle_to_s3(bundle: Path, parsed_location: urllib.parse.ParseResult) -> None:
    bucket = parsed_location.netloc
    remote_dir = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed_location.path)))
    remote_path = str(remote_dir / bundle.name)[1:]  # Remove initial /

    Env.get().console_logger.info(f'Uploading snapshot to: {bucket}:{remote_path}')

    s3_client = Env.get().get_client('s3')

    upload_args = {'ChecksumAlgorithm': 'SHA256'}

    progress_bar = tqdm(
        desc='Uploading',
        total=bundle.stat().st_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,  # Update every iteration (should be fine for an upload)
        ncols=100,
        file=sys.stdout,
    )
    s3_client.upload_file(
        Filename=str(bundle), Bucket=bucket, Key=remote_path, ExtraArgs=upload_args, Callback=progress_bar.update
    )


def pull_replica(dest_path: str, src_tbl_uri: str, version: int | None = None) -> pxt.Table:
    clone_request = ReplicateRequest(table_uri=PxtUri(src_tbl_uri), version=version)
    response = requests.post(PIXELTABLE_API_URL, json=clone_request.model_dump_json(), headers=_api_headers())
    if response.status_code != 200:
        raise excs.Error(f'Error cloning snapshot: {response.text}')
    clone_response = ReplicateResponse.model_validate_json(response.json())
    primary_tbl_additional_md = clone_response.md[0].tbl_md.additional_md
    bundle_uri = str(clone_response.destination_uri)
    bundle_filename = primary_tbl_additional_md['datafile']
    parsed_location = urllib.parse.urlparse(bundle_uri)
    if parsed_location.scheme == 's3':
        bundle_path = _download_bundle_from_s3(parsed_location, bundle_filename)
    elif parsed_location.scheme == 'https':
        bundle_path = TempStore.create_path()
        _download_from_presigned_url(url=parsed_location.geturl(), output_path=bundle_path)
    else:
        raise excs.Error(f'Unexpected response from server: unsupported bundle uri: {bundle_uri}')

    restorer = TableRestorer(
        dest_path,
        {'pxt_version': pxt.__version__, 'pxt_md_version': clone_response.pxt_md_version, 'md': clone_response.md},
    )
    tbl = restorer.restore(bundle_path)
    Env.get().console_logger.info(f'Created local replica {tbl._path()!r} from URI: {src_tbl_uri}')
    return tbl


def _download_bundle_from_s3(parsed_location: urllib.parse.ParseResult, bundle_filename: str) -> Path:
    bucket = parsed_location.netloc
    remote_dir = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed_location.path)))
    remote_path = str(remote_dir / bundle_filename)[1:]  # Remove initial /

    Env.get().console_logger.info(f'Downloading snapshot from: {bucket}:{remote_path}')

    s3_client = Env.get().get_client('s3')

    obj = s3_client.head_object(Bucket=bucket, Key=remote_path)  # Check if the object exists
    bundle_size = obj['ContentLength']

    bundle_path = TempStore.create_path()
    progress_bar = tqdm(
        desc='Downloading',
        total=bundle_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        ncols=100,
        file=sys.stdout,
    )
    s3_client.download_file(Bucket=bucket, Key=remote_path, Filename=str(bundle_path), Callback=progress_bar.update)
    return bundle_path


def _create_retry_session(
    max_retries: int = 3, backoff_factor: float = 1.0, status_forcelist: Optional[list] = None
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

    session = _create_retry_session(max_retries=max_retries)
    try:
        with (
            open(file_path, 'rb') as f,
            tqdm.wrapattr(
                f,
                method='read',
                total=file_size,
                desc='Uploading',
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,  # Update every iteration (should be fine for an upload)
                ncols=100,
                file=sys.stdout,
            ) as file_with_progress,
        ):
            response = session.put(
                url,
                data=file_with_progress,
                headers=headers,
                timeout=(60, 1800),  # 60 seconds to connect and 300 seconds for server response
            )
            response.raise_for_status()
            return response
    finally:
        session.close()


def _download_from_presigned_url(
    url: str, output_path: Path, headers: Optional[dict[str, str]] = None, max_retries: int = 3
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
        progress_bar = tqdm(
            desc='Downloading',
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            ncols=100,
            file=sys.stdout,
        )
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
    finally:
        session.close()


def delete_replica(dest_path: str, version: int | None = None) -> None:
    """Delete cloud replica"""
    delete_request = DeleteRequest(table_uri=PxtUri(dest_path), version=version)
    response = requests.post(PIXELTABLE_API_URL, json=delete_request.model_dump_json(), headers=_api_headers())
    if response.status_code != 200:
        raise excs.Error(f'Error deleting replica: {response.text}')


def _api_headers() -> dict[str, str]:
    headers = {'Content-Type': 'application/json'}
    api_key = Env.get().pxt_api_key
    if api_key is not None:
        headers['X-api-key'] = api_key
    return headers
