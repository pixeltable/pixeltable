import sys
import urllib.parse
import urllib.request
from pathlib import Path

import requests
from tqdm import tqdm

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.metadata.schema import FullTableMd
from pixeltable.utils import sha256sum

from .packager import TablePackager

# These URLs are abstracted out for now, but will be replaced with actual (hard-coded) URLs once the
# pixeltable.com URLs are available.

PIXELTABLE_API_URL = 'https://internal-api.pixeltable.com'


def publish_snapshot(dest_tbl_uri: str, src_tbl: pxt.Table) -> str:
    packager = TablePackager(src_tbl, additional_md={'table_uri': dest_tbl_uri})
    request_json = packager.md | {'operation_type': 'publish_snapshot'}
    headers_json = {'X-api-key': Env.get().pxt_api_key, 'Content-Type': 'application/json'}
    response = requests.post(PIXELTABLE_API_URL, json=request_json, headers=headers_json)
    if response.status_code != 200:
        raise excs.Error(f'Error publishing snapshot: {response.text}')
    response_json = response.json()
    if not isinstance(response_json, dict) or response_json.get('destination') != 's3':
        raise excs.Error(f'Error publishing snapshot: unexpected response from server.\n{response_json}')
    upload_id = response_json['upload_id']
    destination_uri = response_json['destination_uri']

    Env.get().console_logger.info(f"Creating a snapshot of '{src_tbl._path}' at: {dest_tbl_uri}")

    bundle = packager.package()

    parsed_location = urllib.parse.urlparse(destination_uri)
    if parsed_location.scheme == 's3':
        _upload_bundle_to_s3(bundle, parsed_location)
    else:
        raise excs.Error(
            f"ERROR publishing snapshot: The destination URI scheme is not supported: '{{destination_uri}}'. "
            "Currently, only S3 destinations (starting with 's3://') are supported for publishing."
        )

    Env.get().console_logger.info('Finalizing snapshot ...')

    finalize_request_json = {
        'operation_type': 'finalize_snapshot',
        'upload_id': upload_id,
        'datafile': bundle.name,
        'size': bundle.stat().st_size,
        'sha256': sha256sum(bundle),  # Generate our own SHA for independent verification
    }
    # TODO: Use Pydantic for validation
    finalize_response = requests.post(PIXELTABLE_API_URL, json=finalize_request_json, headers=headers_json)
    if finalize_response.status_code != 200:
        raise excs.Error(f'Error finalizing snapshot: {finalize_response.text}')
    finalize_response_json = finalize_response.json()
    if not isinstance(finalize_response_json, dict) or 'confirmed_table_uri' not in finalize_response_json:
        raise excs.Error(f'Error finalizing snapshot: unexpected response from server.\n{finalize_response_json}')

    confirmed_tbl_uri = finalize_response_json['confirmed_table_uri']
    Env.get().console_logger.info(f'The published snapshot is now available at: {confirmed_tbl_uri}')
    return confirmed_tbl_uri


def clone_snapshot(dest_tbl_uri: str) -> list[FullTableMd]:
    headers_json = {'X-api-key': Env.get().pxt_api_key, 'Content-Type': 'application/json'}
    clone_request_json = {'operation_type': 'clone_snapshot', 'table_uri': dest_tbl_uri}
    response = requests.post(PIXELTABLE_API_URL, json=clone_request_json, headers=headers_json)
    if response.status_code != 200:
        raise excs.Error(f'Error cloning snapshot: {response.text}')
    response_json = response.json()
    if not isinstance(response_json, dict) or 'table_uri' not in response_json:
        raise excs.Error(f'Unexpected response from server.\n{response_json}')
    return [FullTableMd.from_dict(t) for t in response_json['md']['tables']]


def _upload_bundle_to_s3(bundle: Path, parsed_location: urllib.parse.ParseResult) -> None:
    from pixeltable.utils.s3 import get_client

    bucket = parsed_location.netloc
    remote_dir = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed_location.path)))
    remote_path = str(remote_dir / bundle.name)[1:]  # Remove initial /

    Env.get().console_logger.info(f'Uploading snapshot to: {bucket}:{remote_path}')

    boto_config = {'max_pool_connections': 5, 'connect_timeout': 15, 'retries': {'max_attempts': 3, 'mode': 'adaptive'}}
    s3_client = get_client(**boto_config)

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
        Filename=str(bundle), Bucket=bucket, Key=str(remote_path), ExtraArgs=upload_args, Callback=progress_bar.update
    )
