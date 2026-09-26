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

A pod template that belongs to a release names it in DIGEST_ENV and BUILD_ID_ENV, and the fetch asks for
that release's archive rather than the database's current one. Every pod the template starts then runs
the same code, whether it starts with the rollout or later, as a restart or a scale-up.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

from pixeltable import exceptions as excs
from pixeltable.service import management_client
from pixeltable.service.db import _validated_db_uri
from pixeltable.service.management_protocol import GetArchiveRequest, GetArchiveResponse
from pixeltable.utils.project import unpacked_digest

PROJECT_SUBDIR = 'project'
FINGERPRINT_FILE = 'fingerprint.json'

# Environment variables rather than flags: a template may name a base image whose fetcher predates them, which
# ignores a variable and would refuse a flag.
DIGEST_ENV = 'PXTCLOUD_ARCHIVE_DIGEST'
BUILD_ID_ENV = 'PXTCLOUD_BUILD_ID'

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


def unpack_project_archive(
    db_uri: str, dest: Path, *, digest: str | None = None, build_id: uuid.UUID | None = None
) -> GetArchiveResponse:
    """Unpack db_uri's project archive into dest; returns the control plane's response.

    digest and build_id name the release to unpack (GetArchiveRequest); without them, the database's current one.
    """
    db_path = _validated_db_uri(db_uri)
    request = GetArchiveRequest(org=db_path.org, db=db_path.db, digest=digest, build_id=build_id)
    response = GetArchiveResponse.model_validate(management_client.api_call(request))
    if digest is not None and response.digest != digest:
        # a control plane that predates GetArchiveRequest.digest serves its current archive, as it did before pods
        # named one; refusing it would take the pod down without pinning anything
        _logger.warning(
            '%s served project %s, not %s, which this pod was started for: its control plane ignores the pin',
            db_path.uri_str,
            response.digest,
            digest,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    # staged next to dest and moved into place, so that dest ends up with exactly the archive's files
    unpacking = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f'.{dest.name}.'))
    archive_path = unpacking / 'project.tar.bz2'
    try:
        # streamed to disk: a project's files may be too large for memory
        with (
            urllib.request.urlopen(response.presigned_url, timeout=_DOWNLOAD_TIMEOUT) as r,
            archive_path.open('wb') as f,
        ):
            shutil.copyfileobj(r, f)

        staged = unpacking / _TARBALL_ROOT
        staged.mkdir()  # an empty archive still unpacks to an empty project
        prefix = f'{_TARBALL_ROOT}/'
        with tarfile.open(archive_path, mode='r:bz2') as tf:
            members: list[tarfile.TarInfo] = []
            for member in tf.getmembers():
                if member.name == _TARBALL_ROOT:
                    continue
                if not member.name.startswith(prefix):
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_DATA_FORMAT,
                        f'{db_path.uri_str} served an archive with a member outside {prefix}: {member.name!r}',
                    )
                member.name = member.name[len(prefix) :]
                if member.islnk() and member.linkname.startswith(prefix):
                    # a hard link points at another member, and that name loses the prefix too
                    member.linkname = member.linkname[len(prefix) :]
                members.append(member)
            # filter='data': refuses a member whose path is outside the directory, and drops ownership bits
            tf.extractall(staged, members=members, filter='data')

        unpacked = unpacked_digest(staged)
        if unpacked != response.digest:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_DATA_FORMAT,
                f'{db_path.uri_str} served project {unpacked}, not {response.digest}',
            )

        if dest.exists():
            shutil.rmtree(dest)
        staged.rename(dest)
    finally:
        shutil.rmtree(unpacking, ignore_errors=True)
    return response


def fetch(db_uri: str, archive_dir: Path, *, digest: str | None = None, build_id: uuid.UUID | None = None) -> bool:
    """Unpack db_uri's project into archive_dir; False if the database has no project yet.

    digest and build_id name the release the pod was started for. That release has a project, so a 404 for it
    raises: its archive is gone, and the pod must not come up without the release's code.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    for delay in _ARCHIVE_FETCH_DELAYS:
        if delay > 0.0:
            time.sleep(delay)
        try:
            response = unpack_project_archive(db_uri, project_dir(archive_dir), digest=digest, build_id=build_id)
        except excs.ExternalServiceError as exc:
            if exc.provider_http_status_code != 404 or digest is not None:
                raise
            continue
        # archive_dir reflects this run alone: an archive served without a fingerprint must not leave
        # an earlier one for the pod to report
        fingerprint_path(archive_dir).unlink(missing_ok=True)
        if response.fingerprint is not None:
            fingerprint_path(archive_dir).write_text(response.fingerprint.model_dump_json(), encoding='utf-8')
        return True
    # the database has no project now, whatever an earlier run left here
    shutil.rmtree(project_dir(archive_dir), ignore_errors=True)
    fingerprint_path(archive_dir).unlink(missing_ok=True)
    return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='pixeltable.service.fetch_archive')
    parser.add_argument('--db', required=True, help='pxt://org:db, the database whose project to unpack')
    parser.add_argument('--archive-dir', type=Path, required=True, help='unpack the project under here')
    parsed = parser.parse_args(argv)
    # an empty value is unset, as for a template that belongs to no release
    digest = os.environ.get(DIGEST_ENV) or None
    build_id = os.environ.get(BUILD_ID_ENV) or None
    if not fetch(
        parsed.db, parsed.archive_dir, digest=digest, build_id=None if build_id is None else uuid.UUID(build_id)
    ):
        # exit 0: a non-zero exit would crash-loop the pod, and a database with no project still serves
        _logger.warning('%s has no project; its udfs cannot be resolved', parsed.db)


if __name__ == '__main__':
    main()
