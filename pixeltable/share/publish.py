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

# These URLs are abstracted out for now, but will be replaced with actual (hard-coded) URLs once the
# pixeltable.com URLs are available.

PIXELTABLE_API_URL = os.environ.get('PIXELTABLE_API_URL', 'https://internal-api.pixeltable.com')


def push_replica(
    dest_tbl_uri: str, src_tbl: pxt.Table, bucket: str | None = None, access: Literal['public', 'private'] = 'private'
) -> str:
    packager = TablePackager(
        src_tbl, additional_md={'table_uri': dest_tbl_uri, 'bucket_name': bucket, 'is_public': access == 'public'}
    )
    request_json = packager.md | {'operation_type': 'publish_snapshot'}
    response = requests.post(PIXELTABLE_API_URL, json=request_json, headers=_api_headers())
    if response.status_code != 200:
        raise excs.Error(f'Error publishing snapshot: {response.text}')
    response_json = response.json()
    if not isinstance(response_json, dict):
        raise excs.Error(f'Error publishing snapshot: unexpected response from server.\n{response_json}')
    upload_id = response_json['upload_id']
    destination_uri = response_json['destination_uri']

    Env.get().console_logger.info(f"Creating a snapshot of '{src_tbl._path()}' at: {dest_tbl_uri}")

    bundle = packager.package()

    parsed_location = urllib.parse.urlparse(destination_uri)
    if parsed_location.scheme == 's3':
        _upload_bundle_to_s3(bundle, parsed_location)
    elif parsed_location.scheme == 'https':
        _upload_to_presigned_url(file_path=bundle, url=parsed_location.geturl())
    else:
        raise excs.Error(f'Unsupported destination: {destination_uri}')

    Env.get().console_logger.info('Finalizing snapshot ...')

    finalize_request_json = {
        'table_uri': dest_tbl_uri,
        'operation_type': 'finalize_snapshot',
        'upload_id': upload_id,
        'datafile': bundle.name,
        'size': bundle.stat().st_size,
        'sha256': sha256sum(bundle),  # Generate our own SHA for independent verification
        'rows': packager.md['row_count'],  # TODO rename rows to row_count once cloud side changes are complete
        'preview_header': packager.md['preview_header'],
        'preview_data': packager.md['preview_data'],
    }
    # TODO: Use Pydantic for validation
    finalize_response = requests.post(PIXELTABLE_API_URL, json=finalize_request_json, headers=_api_headers())
    if finalize_response.status_code != 200:
        raise excs.Error(f'Error finalizing snapshot: {finalize_response.text}')
    finalize_response_json = finalize_response.json()
    if not isinstance(finalize_response_json, dict) or 'confirmed_table_uri' not in finalize_response_json:
        raise excs.Error(f'Error finalizing snapshot: unexpected response from server.\n{finalize_response_json}')

    confirmed_tbl_uri = finalize_response_json['confirmed_table_uri']
    Env.get().console_logger.info(f'The published snapshot is now available at: {confirmed_tbl_uri}')
    return confirmed_tbl_uri


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


def pull_replica(dest_path: str, src_tbl_uri: str) -> pxt.Table:
    clone_request_json = {'operation_type': 'clone_snapshot', 'table_uri': src_tbl_uri}
    response = requests.post(PIXELTABLE_API_URL, json=clone_request_json, headers=_api_headers())
    if response.status_code != 200:
        raise excs.Error(f'Error cloning snapshot: {response.text}')
    response_json = response.json()
    if not isinstance(response_json, dict) or 'table_uri' not in response_json:
        raise excs.Error(f'Error cloning shapshot: unexpected response from server.\n{response_json}')

    primary_tbl_additional_md = response_json['md']['tables'][0]['table_md']['additional_md']
    bundle_uri = response_json['destination_uri']
    bundle_filename = primary_tbl_additional_md['datafile']
    parsed_location = urllib.parse.urlparse(bundle_uri)
    if parsed_location.scheme == 's3':
        bundle_path = _download_bundle_from_s3(parsed_location, bundle_filename)
    elif parsed_location.scheme == 'https':
        bundle_path = TempStore.create_path()
        _download_from_presigned_url(url=parsed_location.geturl(), output_path=bundle_path)
    else:
        raise excs.Error(f'Unexpected response from server: unsupported bundle uri: {bundle_uri}')

    restorer = TableRestorer(dest_path, response_json)
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


def delete_replica(dest_path: str) -> None:
    """Delete cloud replica"""
    delete_request_json = {'operation_type': 'delete_snapshot', 'table_uri': dest_path}
    response = requests.post(PIXELTABLE_API_URL, json=delete_request_json, headers=_api_headers())
    if response.status_code != 200:
        raise excs.Error(f'Error deleting replica: {response.text}')
    response_json = response.json()
    if not isinstance(response_json, dict) or 'table_uri' not in response_json:
        raise excs.Error(f'Error deleting replica: unexpected response from server.\n{response_json}')


def _api_headers() -> dict[str, str]:
    headers = {'Content-Type': 'application/json'}
    api_key = Env.get().pxt_api_key
    if api_key is not None:
        headers['X-api-key'] = api_key
    return headers
