"""The init container's side of the archive handoff.

fetch_archive runs before the container that serves a database, so everything here is a startup path:
a regression in it takes the pod down (or, worse, brings it up serving nothing) and shows up only in a
cluster. The three branches are what the pod's behaviour hangs on -- an archive that unpacks, a
database that has no project yet, and a failure that is neither.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from pixeltable import exceptions as excs
from pixeltable.service import fetch_archive
from pixeltable.utils.project import project_fingerprint

_DB_URI = 'pxt://acme:main'


def _served(project_root: Path) -> mock.Mock:
    """A GetArchive response for project_root, as unpack_project_archive returns it."""
    return mock.Mock(fingerprint=project_fingerprint(project_root, None))


def _not_found() -> excs.ExternalServiceError:
    return excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'no archive', status_code=404)


class TestFetchArchive:
    def test_unpacks_the_project_and_records_the_fingerprint(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / 'archive'
        source = tmp_path / 'src'
        source.mkdir()
        (source / 'app.py').write_text('x = 1\n', encoding='utf-8')

        def _unpack(db_uri: str, dest: Path) -> mock.Mock:
            dest.mkdir(parents=True)
            (dest / 'app.py').write_text('x = 1\n', encoding='utf-8')
            return _served(source)

        with mock.patch.object(fetch_archive, 'unpack_project_archive', side_effect=_unpack) as unpack:
            assert fetch_archive.fetch(_DB_URI, archive_dir) is True

        # unpacked where the pod is told to look, not merely somewhere under the archive directory
        assert unpack.call_args.args[1] == fetch_archive.project_dir(archive_dir)
        assert (fetch_archive.project_dir(archive_dir) / 'app.py').is_file()
        # the fingerprint is what a service pod reports back, so it has to survive to disk intact
        recorded = json.loads(fetch_archive.fingerprint_path(archive_dir).read_text(encoding='utf-8'))
        assert recorded == json.loads(project_fingerprint(source, None).model_dump_json())

    def test_a_database_with_no_project_is_not_a_failure(self, tmp_path: Path) -> None:
        """A 404 that outlasts the retries means the database has no project yet, which pods must serve."""
        archive_dir = tmp_path / 'archive'
        with (
            mock.patch.object(fetch_archive, 'unpack_project_archive', side_effect=_not_found()) as unpack,
            mock.patch.object(fetch_archive.time, 'sleep') as sleep,
        ):
            assert fetch_archive.fetch(_DB_URI, archive_dir) is False

        # every delay is spent before concluding it: an archive uploaded moments ago 404s too
        assert unpack.call_count == len(fetch_archive._ARCHIVE_FETCH_DELAYS)
        assert sleep.call_count == len(fetch_archive._ARCHIVE_FETCH_DELAYS) - 1
        # nothing unpacked, and no stale fingerprint left claiming otherwise
        assert not fetch_archive.project_dir(archive_dir).exists()
        assert not fetch_archive.fingerprint_path(archive_dir).exists()

    def test_a_later_attempt_still_wins(self, tmp_path: Path) -> None:
        """The retries exist for an archive that is not readable yet, not only for one that is missing."""
        archive_dir = tmp_path / 'archive'
        source = tmp_path / 'src'
        source.mkdir()

        attempts = 0

        def _unpack(db_uri: str, dest: Path) -> mock.Mock:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise _not_found()
            dest.mkdir(parents=True)
            return _served(source)

        with (
            mock.patch.object(fetch_archive, 'unpack_project_archive', side_effect=_unpack),
            mock.patch.object(fetch_archive.time, 'sleep'),
        ):
            assert fetch_archive.fetch(_DB_URI, archive_dir) is True
        assert attempts == 3
        assert fetch_archive.fingerprint_path(archive_dir).is_file()

    @pytest.mark.parametrize('status_code', [401, 500])
    def test_other_failures_propagate(self, tmp_path: Path, status_code: int) -> None:
        """Only a 404 means 'no project'. Anything else must fail the init container rather than
        leave the pod up serving a database whose project could not be read."""
        archive_dir = tmp_path / 'archive'
        err = excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'nope', status_code=status_code)
        with (
            mock.patch.object(fetch_archive, 'unpack_project_archive', side_effect=err) as unpack,
            mock.patch.object(fetch_archive.time, 'sleep'),
            pytest.raises(excs.ExternalServiceError),
        ):
            fetch_archive.fetch(_DB_URI, archive_dir)
        assert unpack.call_count == 1  # not retried
