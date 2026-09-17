"""Unpack a hosted database's project archive, ahead of the pod that serves it.

The container that serves a database cannot fetch its own archive. The image installs the project's
dependencies but not the project itself (`uv sync --no-install-project`), because the image context
carries only the manifests -- so for a project whose own package is pixeltable, the pixeltable that
would do the fetching is the thing being fetched. This module runs ahead of it, in an init container
on the base image, which carries a pixeltable of its own regardless of what the project pins.

Layout under --archive-dir, which both containers mount:

    project/          the unpacked project; absent when the database has no project yet
    fingerprint.json  what the control plane served the archive as

Both are named to a pod by their own path, so this module is the only place that decides where they
go. The fingerprint is written down rather than re-fetched: a pod reports the archive it actually
loaded, and a second GetArchive call could answer with a different one.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from pixeltable import exceptions as excs
from pixeltable.service.db import unpack_project_archive

PROJECT_SUBDIR = 'project'
FINGERPRINT_FILE = 'fingerprint.json'

# A database exists before `pxt db update` first gives it a project, and an archive uploaded moments
# ago may not be readable yet; both look like a 404 here, so the delays cover the second case before
# the first is concluded.
_ARCHIVE_FETCH_DELAYS = (0.0, 1.0, 2.0, 4.0)

_logger = logging.getLogger('pixeltable')


def project_dir(archive_dir: Path) -> Path:
    return archive_dir / PROJECT_SUBDIR


def fingerprint_path(archive_dir: Path) -> Path:
    return archive_dir / FINGERPRINT_FILE


def fetch(db_uri: str, archive_dir: Path) -> bool:
    """Unpack db_uri's project into archive_dir; False if it has no project yet."""
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
        # exit 0 regardless: a non-zero exit crash-loops the pod, and a database with no project is
        # expected to serve
        _logger.warning('%s has no project; udfs it defines cannot be resolved', parsed.db)


if __name__ == '__main__':
    main()
