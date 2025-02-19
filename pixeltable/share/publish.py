import dataclasses
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

import pixeltable as pxt
from pixeltable import exceptions as excs, metadata
from pixeltable.env import Env
from pixeltable.utils import sha256sum

from .packager import TablePackager

_PUBLISH_URL = 'https://cf4ggxh3bgocx65j5wwbdbk2iu0bawoi.lambda-url.us-east-1.on.aws/?debug=false'


def publish_snapshot(dest_uri: str, src_tbl: pxt.Table) -> None:
    request_json = {
        'pxt_version': pxt.__version__,
        'pxt_schema_version': metadata.VERSION,
        'table_uri': dest_uri,
        'md': {
            'table_md': dataclasses.asdict(src_tbl._tbl_version._create_tbl_md()),
            'table_version_md': dataclasses.asdict(src_tbl._tbl_version._create_version_md(datetime.now().timestamp())),
            'table_schema_version_md': dataclasses.asdict(src_tbl._tbl_version._create_schema_version_md(0)),
        },
    }
    headers_json = {'X-api-key': Env.get().pxt_api_key}

    response = requests.post(_PUBLISH_URL, json=request_json, headers=headers_json)
    if response.status_code != 200:
        raise excs.Error(f'Error publishing snapshot: {response.text}')
    response_json = response.json()
    if not isinstance(response_json, dict) or 's3location' not in response_json:
        raise excs.Error(f'Error publishing snapshot: invalid response.\n{response_json}')
    location = response_json['s3location']

    Env.get().console_logger.info(f"Creating a snapshot of '{src_tbl._path}' at: {dest_uri}")

    packager = TablePackager(src_tbl)
    bundle = packager.package()

    parsed_location = urllib.parse.urlparse(location)
    if parsed_location.scheme == 's3':
        _upload_bundle_to_s3(bundle, parsed_location)
    else:
        raise excs.Error(f'Unsupported destination: {location}')

    Env.get().console_logger.info(f'Finalizing snapshot ...')


def _upload_bundle_to_s3(bundle: Path, parsed_location: urllib.parse.ParseResult) -> None:
    from pixeltable.utils.s3 import get_client

    bucket = parsed_location.netloc
    remote_dir = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed_location.path)))
    remote_path = str(remote_dir / bundle.name)[1:]  # Remove initial /

    Env.get().console_logger.info(f'Uploading snapshot to: {bucket}:{remote_path}')

    boto_config = {
        'max_pool_connections': 5,
        'connect_timeout': 86400,
        'retries': {'max_attempts': 3, 'mode': 'adaptive'},
    }
    s3_client = get_client(**boto_config)

    upload_args = {'ChecksumAlgorithm': 'SHA256'}

    progress_bar = tqdm(
        desc=f'Uploading',
        total=bundle.stat().st_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        ncols=100,
        file=sys.stdout,
    )
    s3_client.upload_file(
        Filename=str(bundle),
        Bucket=bucket,
        Key=str(remote_path),
        ExtraArgs=upload_args,
        Callback=progress_bar.update
    )

    # response = s3_client.get_object(Bucket=bucket, Key=str(remote_path))
