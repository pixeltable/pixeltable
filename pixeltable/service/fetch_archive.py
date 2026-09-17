"""Unpack a hosted database's project archive before its pod starts.

A database's serving container cannot fetch its own archive. The image installs the project's
dependencies but not the project itself (`uv sync --no-install-project`), because the image context
has only the manifests. For pixeltable's own repo, deployed as a project, pixeltable would have to
fetch itself. This module runs first instead, in an init container on the base image, which has its
own pixeltable.

Layout under --archive-dir, mounted by both containers:

    project/          the unpacked project; absent until the database has one
    fingerprint.json  the control plane's fingerprint for that archive

The fingerprint is written to disk rather than re-fetched: a pod reports which archive it loaded,
and a second GetArchive call could return a different one.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from pixeltable import exceptions as excs
from pixeltable.service import management_client
from pixeltable.service.db import _validated_db_uri
from pixeltable.service.management_protocol import GetArchiveRequest, GetArchiveResponse
from pixeltable.utils.project import unpacked_digest

PROJECT_SUBDIR = 'project'
FINGERPRINT_FILE = 'fingerprint.json'

_DOWNLOAD_TIMEOUT = 300
# the archive's own top-level directory, inside the tarball
_TARBALL_ROOT = 'project'

# get_archive returns 404 both for a database with no project and for an archive uploaded moments ago
# that is not readable yet. The retries tell the two apart.
_ARCHIVE_FETCH_DELAYS = (0.0, 1.0, 2.0, 4.0)

_logger = logging.getLogger('pixeltable')


def project_dir(archive_dir: Path) -> Path:
    return archive_dir / PROJECT_SUBDIR


def fingerprint_path(archive_dir: Path) -> Path:
    return archive_dir / FINGERPRINT_FILE


def unpack_project_archive(db_uri: str, dest: Path) -> GetArchiveResponse:
    """Unpack db_uri's project archive into dest, and return what the control plane served it as."""
    db_path = _validated_db_uri(db_uri)
    response = GetArchiveResponse.model_validate(
        management_client.api_call(GetArchiveRequest(org=db_path.org, db=db_path.db))
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    # unpacked next to dest and moved into place, so that dest never holds a file the archive dropped
    unpacking = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f'.{dest.name}.'))
    archive_path = unpacking / 'project.tar.bz2'
    try:
        # streamed to disk: a project may select files too large to hold in memory
        with (
            urllib.request.urlopen(response.presigned_url, timeout=_DOWNLOAD_TIMEOUT) as r,
            archive_path.open('wb') as f,
        ):
            shutil.copyfileobj(r, f)

        staged = unpacking / _TARBALL_ROOT
        staged.mkdir()  # an archive holding no files still unpacks to an empty project
        prefix = f'{_TARBALL_ROOT}/'
        with tarfile.open(archive_path, mode='r:bz2') as tf:
            members: list[tarfile.TarInfo] = []
            for member in tf.getmembers():
                if member.name == _TARBALL_ROOT:
                    continue
                if not member.name.startswith(prefix):
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_DATA_FORMAT,
                        f'{db_path.uri_str} serves an archive holding {member.name!r}, which is outside {prefix}',
                    )
                member.name = member.name[len(prefix) :]
                if member.islnk() and member.linkname.startswith(prefix):
                    # a hard link points at another member, and that name loses the prefix too
                    member.linkname = member.linkname[len(prefix) :]
                members.append(member)
            # filter='data': refuses a member naming a path outside the directory, and drops ownership bits
            tf.extractall(staged, members=members, filter='data')

        unpacked = unpacked_digest(staged)
        if unpacked != response.digest:
            # what arrived is not what the control plane named, whatever it named
            raise excs.RequestError(
                excs.ErrorCode.INVALID_DATA_FORMAT,
                f'{db_path.uri_str} served an archive holding project {unpacked}, not {response.digest}',
            )

        if dest.exists():
            shutil.rmtree(dest)
        staged.rename(dest)
    finally:
        shutil.rmtree(unpacking, ignore_errors=True)
    return response


def fetch(db_uri: str, archive_dir: Path) -> bool:
    """Unpack db_uri's project into archive_dir; False if the database has no project yet."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    for delay in _ARCHIVE_FETCH_DELAYS:
        if delay > 0.0:
            time.sleep(delay)
        try:
            response = unpack_project_archive(db_uri, project_dir(archive_dir))
        except excs.ExternalServiceError as exc:
            if exc.provider_http_status_code != 404:
                raise
            continue
        if response.fingerprint is not None:
            fingerprint_path(archive_dir).write_text(response.fingerprint.model_dump_json(), encoding='utf-8')
        return True
    return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='pixeltable.service.fetch_archive')
    parser.add_argument('--db', required=True, help='pxt://org:db, the database whose project to unpack')
    parser.add_argument('--archive-dir', type=Path, required=True, help='unpack the project under here')
    parsed = parser.parse_args(argv)
    if not fetch(parsed.db, parsed.archive_dir):
        # exit 0: a non-zero exit would crash-loop the pod, and a database with no project still serves
        _logger.warning('%s has no project; its udfs cannot be resolved', parsed.db)


if __name__ == '__main__':
    main()
