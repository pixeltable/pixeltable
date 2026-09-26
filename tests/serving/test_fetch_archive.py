"""The init container's side of the archive handoff: unpacking an archive, and the branches around it.

fetch_archive runs before a database's serving container, so everything here is a startup path: a
regression takes the pod down, or brings it up serving nothing, and shows up only in a cluster. The
pod's behavior depends on three branches: an archive that unpacks, a database with no project, and
any other failure.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from pixeltable import exceptions as excs
from pixeltable.service import fetch_archive
from pixeltable.service.management_protocol import GetArchiveRequest
from pixeltable.utils.project import package_project_archive, project_fingerprint, unpacked_digest

from ..utils import pxt_raises

_DB_URI = 'pxt://acme:main'
_BUILD_ID = uuid.UUID('6f1c2a4e-0000-4000-8000-000000000001')


def _served(project_root: Path) -> mock.Mock:
    """A GetArchive response for project_root."""
    return mock.Mock(fingerprint=project_fingerprint(project_root, None))


def _not_found() -> excs.ExternalServiceError:
    return excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'no archive', status_code=404)


def _control_plane(project: Path, sent: list[GetArchiveRequest]) -> Any:
    """api_call for a control plane serving project's archive, whatever the request names; records each request."""
    (project / 'app.py').write_text('x = 1\n')
    packaged = package_project_archive(project)
    fingerprint = project_fingerprint(project, None)

    def _api_call(request: GetArchiveRequest) -> dict[str, Any]:
        sent.append(request)
        return {
            'presigned_url': packaged.path.as_uri(),
            'digest': fingerprint.archive_digest(),
            'fingerprint': fingerprint.model_dump(mode='json'),
        }

    return _api_call


class TestFetchArchive:
    def test_unpacks_the_project_and_records_the_fingerprint(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / 'archive'
        source = tmp_path / 'src'
        source.mkdir()
        (source / 'app.py').write_text('x = 1\n', encoding='utf-8')

        def _unpack(db_uri: str, dest: Path, **_release: object) -> mock.Mock:
            dest.mkdir(parents=True)
            (dest / 'app.py').write_text('x = 1\n', encoding='utf-8')
            return _served(source)

        with mock.patch.object(fetch_archive, 'unpack_project_archive', side_effect=_unpack) as unpack:
            assert fetch_archive.fetch(_DB_URI, archive_dir) is True

        # unpacked at project_dir(), not just somewhere under archive_dir
        assert unpack.call_args.args[1] == fetch_archive.project_dir(archive_dir)
        assert (fetch_archive.project_dir(archive_dir) / 'app.py').is_file()
        # a service pod reports this fingerprint back, so it must survive to disk intact
        recorded = json.loads(fetch_archive.fingerprint_path(archive_dir).read_text(encoding='utf-8'))
        assert recorded == json.loads(project_fingerprint(source, None).model_dump_json())

    def test_a_database_with_no_project_is_not_a_failure(self, tmp_path: Path) -> None:
        """A 404 that outlasts the retries means the database has no project yet. Pods must still serve it."""
        archive_dir = tmp_path / 'archive'
        # an earlier run's project and fingerprint, which this one must not leave behind
        fetch_archive.project_dir(archive_dir).mkdir(parents=True)
        (fetch_archive.project_dir(archive_dir) / 'stale.py').write_text('x = 1\n', encoding='utf-8')
        fetch_archive.fingerprint_path(archive_dir).write_text('{}', encoding='utf-8')

        with (
            mock.patch.object(fetch_archive, 'unpack_project_archive', side_effect=_not_found()) as unpack,
            mock.patch.object(fetch_archive.time, 'sleep') as sleep,
        ):
            assert fetch_archive.fetch(_DB_URI, archive_dir) is False

        # every delay is spent first: an archive uploaded moments ago also 404s
        assert unpack.call_count == len(fetch_archive._ARCHIVE_FETCH_DELAYS)
        assert sleep.call_count == len(fetch_archive._ARCHIVE_FETCH_DELAYS) - 1
        # nothing unpacked, and no leftover fingerprint
        assert not fetch_archive.project_dir(archive_dir).exists()
        assert not fetch_archive.fingerprint_path(archive_dir).exists()

    def test_a_later_attempt_still_wins(self, tmp_path: Path) -> None:
        """The retries cover an archive that is not readable yet, not only one that is missing."""
        archive_dir = tmp_path / 'archive'
        source = tmp_path / 'src'
        source.mkdir()

        attempts = 0

        def _unpack(db_uri: str, dest: Path, **_release: object) -> mock.Mock:
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

    def test_an_archive_without_a_fingerprint_clears_the_old_one(self, tmp_path: Path) -> None:
        """GetArchiveResponse.fingerprint is optional, and a pod reports whatever is on disk."""
        archive_dir = tmp_path / 'archive'
        fetch_archive.fingerprint_path(archive_dir).parent.mkdir(parents=True)
        fetch_archive.fingerprint_path(archive_dir).write_text('{}', encoding='utf-8')

        def _unpack(db_uri: str, dest: Path, **_release: object) -> mock.Mock:
            dest.mkdir(parents=True, exist_ok=True)
            return mock.Mock(fingerprint=None)

        with mock.patch.object(fetch_archive, 'unpack_project_archive', side_effect=_unpack):
            assert fetch_archive.fetch(_DB_URI, archive_dir) is True
        assert not fetch_archive.fingerprint_path(archive_dir).exists()

    @pytest.mark.parametrize('status_code', [401, 500])
    def test_other_failures_propagate(self, tmp_path: Path, status_code: int) -> None:
        """Only a 404 means 'no project'. Anything else fails the init container, instead of bringing the
        pod up with no project."""
        archive_dir = tmp_path / 'archive'
        err = excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'nope', status_code=status_code)
        with (
            mock.patch.object(fetch_archive, 'unpack_project_archive', side_effect=err) as unpack,
            mock.patch.object(fetch_archive.time, 'sleep'),
            pxt_raises(excs.ErrorCode.PROVIDER_ERROR, match='nope'),
        ):
            fetch_archive.fetch(_DB_URI, archive_dir)
        assert unpack.call_count == 1  # not retried


class TestPinnedRelease:
    """A pod template names the release it belongs to, and the fetch asks for that release's archive.

    Without the pin, a pod that starts after its database moved on (a restart, a scale-up, a surge pod) loads
    another release's code than the one its image and its sibling pods run.
    """

    def test_the_template_pin_reaches_the_request(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        sent: list[GetArchiveRequest] = []
        project = tmp_path / 'project'
        project.mkdir()
        monkeypatch.setattr(fetch_archive.management_client, 'api_call', _control_plane(project, sent))
        digest = project_fingerprint(project, None).archive_digest()
        monkeypatch.setenv(fetch_archive.DIGEST_ENV, digest)
        monkeypatch.setenv(fetch_archive.BUILD_ID_ENV, str(_BUILD_ID))
        archive_dir = tmp_path / 'archive'

        fetch_archive.main(['--db', _DB_URI, '--archive-dir', str(archive_dir)])

        (request,) = sent
        # the body api_call sends
        body = json.loads(request.model_dump_json(by_alias=True))
        assert (body['digest'], body['build_id']) == (digest, str(_BUILD_ID))
        assert (fetch_archive.project_dir(archive_dir) / 'app.py').is_file()

    @pytest.mark.parametrize('value', [None, ''])
    def test_a_template_with_no_release_asks_for_the_current_archive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
    ) -> None:
        """As a template from before pins does, or one for a database created with no project yet."""
        for name in (fetch_archive.DIGEST_ENV, fetch_archive.BUILD_ID_ENV):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        sent: list[GetArchiveRequest] = []

        def _api_call(request: GetArchiveRequest) -> dict[str, Any]:
            sent.append(request)
            raise _not_found()

        monkeypatch.setattr(fetch_archive.management_client, 'api_call', _api_call)
        with mock.patch.object(fetch_archive.time, 'sleep'):
            # no project yet, which is still no failure
            fetch_archive.main(['--db', _DB_URI, '--archive-dir', str(tmp_path / 'archive')])

        assert len(sent) == len(fetch_archive._ARCHIVE_FETCH_DELAYS)
        assert all((r.digest, r.build_id) == (None, None) for r in sent)

    def test_a_release_whose_archive_is_gone_fails_the_pod(self, tmp_path: Path) -> None:
        """A release has a project, so its 404 is not 'no project yet': coming up without the code is worse."""
        with (
            mock.patch.object(fetch_archive, 'unpack_project_archive', side_effect=_not_found()) as unpack,
            mock.patch.object(fetch_archive.time, 'sleep'),
            pxt_raises(excs.ErrorCode.PROVIDER_ERROR, match='no archive'),
        ):
            fetch_archive.fetch(_DB_URI, tmp_path / 'archive', digest='d' * 64, build_id=_BUILD_ID)
        # not retried: no upload is under way for an archive a template already names
        assert unpack.call_count == 1

    def test_a_control_plane_that_ignores_the_pin_still_serves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One that predates the pin serves its current archive, as every pod got before. Refusing that would take
        the pod down during a rollback of the control plane, and pin nothing."""
        sent: list[GetArchiveRequest] = []
        project = tmp_path / 'project'
        project.mkdir()
        monkeypatch.setattr(fetch_archive.management_client, 'api_call', _control_plane(project, sent))
        unpacked = tmp_path / 'unpacked'

        with mock.patch.object(fetch_archive, '_logger') as logger:
            response = fetch_archive.unpack_project_archive(_DB_URI, unpacked, digest='d' * 64, build_id=_BUILD_ID)

        # verified against the digest it was served under, not the one asked for
        assert unpacked_digest(unpacked) == response.digest
        logger.warning.assert_called_once()

    def test_a_field_from_a_newer_fetcher_is_ignored(self) -> None:
        """pixeltable-cloud parses requests with the GetArchiveRequest of the pixeltable it pins. That is how a pin
        reaches a control plane from before pins, and a field added later must reach this one as harmlessly."""
        body = {'operation_type': 'get_archive', 'db': 'main', 'digest': 'd' * 64, 'added_later': 1}
        request = GetArchiveRequest.model_validate(body)
        assert (request.digest, request.build_id) == ('d' * 64, None)


class TestUnpack:
    def test_round_trip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unpacking an archive yields the project it was packaged from, whatever shapes its files take.

        unpack_project_archive() checks the unpacked project against the digest the caller served it, so
        serving this project's own digest makes the call itself the assertion.
        """
        project = tmp_path / 'project'
        (project / 'pkg').mkdir(parents=True)
        (project / 'app.py').write_text('x = 1\n')
        (project / 'pkg' / 'mod.py').write_text('y = 2\n')
        (project / 'link.py').symlink_to('app.py')
        os.link(project / 'app.py', project / 'hard.py')

        packaged = package_project_archive(project)
        fingerprint = project_fingerprint(project, None)
        served = {
            'presigned_url': packaged.path.as_uri(),
            'digest': fingerprint.archive_digest(),
            'fingerprint': fingerprint.model_dump(mode='json'),
        }
        monkeypatch.setattr('pixeltable.service.fetch_archive.management_client.api_call', lambda request: served)

        unpacked = tmp_path / 'unpacked'
        fetch_archive.unpack_project_archive('pxt://acme:main', unpacked)

        assert unpacked_digest(unpacked) == fingerprint.archive_digest()
        assert (unpacked / 'pkg' / 'mod.py').read_text() == 'y = 2\n'
        assert (unpacked / 'link.py').is_symlink() and os.readlink(unpacked / 'link.py') == 'app.py'
        assert (unpacked / 'hard.py').read_text() == 'x = 1\n'
