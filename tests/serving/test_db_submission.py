"""The client's side of a hosted database update: prepare, upload, submit, and wait on the receipt.

A fake control plane answers the management API, so these check what the client sends and how it reads the
answers, not what a real control plane does with them.
"""

from __future__ import annotations

import base64
import hashlib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import requests

from pixeltable import exceptions as excs, metadata
from pixeltable.catalog import Path as PxtPath
from pixeltable.config import Config
from pixeltable.service import management_client, receipts
from pixeltable.service.db import db_build_image, db_diff, db_update
from pixeltable.service.db_md import DatabaseResources, DatabaseStatus
from pixeltable.service.management_protocol import (
    BlobUpload,
    DatabaseReport,
    DatabaseTarget,
    ExpectedGenerations,
    GetDbRequest,
    GetDbResponse,
    GetReceiptsRequest,
    GetReceiptsResponse,
    PrepareUpdateRequest,
    PrepareUpdateResponse,
    SetSecretRequest,
    SubmitUpdateRequest,
    SubmitUpdateResponse,
)
from pixeltable_cli.types import DbPlan, DbState, GenerationReceipt, ReceiptError, ReceiptOutcome, ResourcePhase

from ..utils import pxt_raises

_DB_URI = 'pxt://acme:main'


class _ControlPlane:
    """Answers the management API for one database whose current generation is `generation`.

    current: the receipt of that generation. drift: how far another client moves the generation after each read.
    """

    def __init__(
        self,
        generation: int,
        settled: GenerationReceipt,
        *,
        in_agreement: bool = False,
        current: GenerationReceipt | None = None,
        drift: int = 0,
        refuse: bool = False,
    ) -> None:
        self.generation = generation
        self.settled = settled
        self.in_agreement = in_agreement
        self.current = current
        self.drift = drift
        self.refuse = refuse
        self.stored: dict[str, bytes] = {}
        self.uploads: list[dict[str, str]] = []
        self.submissions: list[SubmitUpdateRequest] = []
        self.receipt_reads = 0

    def api_call(self, request: Any, credential: Any = None) -> dict[str, Any]:
        if isinstance(request, PrepareUpdateRequest):
            resolution = 'up_to_date' if self.in_agreement else 'update_additive'
            plan = DbPlan(db_uri=_DB_URI, exists=True, state='AVAILABLE', resolution=resolution)
            upload = None
            if request.blob is not None and request.blob.sha256 not in self.stored:
                upload = BlobUpload(url=f'https://store.example.com/{request.blob.sha256}')
            response = PrepareUpdateResponse(
                generations=ExpectedGenerations(database=self.generation),
                report=self._report(),
                plan=plan,
                upload=upload,
            ).model_dump(mode='json')
            self.generation += self.drift
            return response
        if isinstance(request, SubmitUpdateRequest):
            self.submissions.append(request)
            if self.refuse:
                raise excs.ConcurrencyError(excs.ErrorCode.CONCURRENT_MODIFICATION, 'Pixeltable Cloud refused this')
            accepted = self.settled.model_copy(update={'outcome': None, 'error': None, 'phase': 'PENDING'})
            return SubmitUpdateResponse(receipts=[accepted]).model_dump(mode='json')
        if isinstance(request, GetReceiptsRequest):
            self.receipt_reads += 1
            return GetReceiptsResponse(receipts=[self.settled]).model_dump(mode='json')
        if isinstance(request, GetDbRequest):
            return GetDbResponse(report=self._report()).model_dump(mode='json')
        raise AssertionError(f'unexpected request {request.operation_type}')

    def _report(self) -> DatabaseReport:
        """A database resized to 4 cpus that still runs on 2."""
        current = DatabaseStatus(resources=DatabaseResources(cpu=2.0), state=DbState.AVAILABLE)
        return DatabaseReport(
            db='main', current=current, target_resources=DatabaseResources(cpu=4.0), receipt=self.current
        )

    def urlopen(self, request: urllib.request.Request, timeout: float) -> Any:
        body = request.data.read()  # type: ignore[union-attr]
        headers = {name.lower(): value for name, value in request.header_items()}
        self.uploads.append(headers)
        self.stored[hashlib.sha256(body).hexdigest()] = body
        return _Closable()


class _Closable:
    def close(self) -> None:
        pass


def _receipt(**fields: Any) -> GenerationReceipt:
    return GenerationReceipt(kind='database', resource_id='db-uuid', generation=8, db='main', **fields)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / 'pixeltable.toml').write_text(f'[[pixeltable.database]]\nname = "{_DB_URI}"\n', encoding='utf-8')
    (tmp_path / 'app.py').write_text('x = 1\n')
    (tmp_path / 'requirements.txt').write_text('pandas\n')
    Config.init(reinit=True, project_root=tmp_path)
    monkeypatch.setattr(receipts, 'POLL_INTERVAL', 0.0)
    return tmp_path


def _serve(monkeypatch: pytest.MonkeyPatch, control_plane: _ControlPlane) -> None:
    monkeypatch.setattr(management_client, 'api_call', control_plane.api_call)
    monkeypatch.setattr(urllib.request, 'urlopen', control_plane.urlopen)


class TestDbSubmission:
    def test_update(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The archive is uploaded once under its own hash, and submitted against the plan's generation."""
        control_plane = _ControlPlane(generation=7, settled=_receipt(outcome=ReceiptOutcome.OBSERVED))
        _serve(monkeypatch, control_plane)

        applied = db_update(_DB_URI, expected_generation=5)
        assert applied.status == 'applied'
        assert [r.observed for r in applied.receipts] == [True]

        (submitted,) = control_plane.submissions
        assert submitted.expected_generations.database == 5, 'the plan the caller confirmed, not a fresh read'
        assert submitted.database_target is not None
        assert 'default_bucket' not in submitted.database_target.model_dump(), 'the control plane owns it'
        assert submitted.blob is not None
        (upload,) = control_plane.uploads
        assert upload['if-none-match'] == '*'
        assert upload['x-amz-checksum-sha256'] == base64.b64encode(bytes.fromhex(submitted.blob.sha256)).decode()
        assert submitted.blob.sha256 in control_plane.stored

        db_update(_DB_URI, wait=False)
        assert len(control_plane.uploads) == 1, 'an unchanged project is not uploaded again'
        assert control_plane.submissions[1].blob == submitted.blob, 'an unchanged project is the same blob'

    def test_upload_network_failure(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed upload is retried once, then reported as a pixeltable error rather than a raw traceback."""
        control_plane = _ControlPlane(generation=7, settled=_receipt(outcome=ReceiptOutcome.OBSERVED))
        _serve(monkeypatch, control_plane)
        attempts: list[urllib.request.Request] = []

        def refuse(request: urllib.request.Request, timeout: float) -> Any:
            attempts.append(request)
            raise urllib.error.URLError('connection reset by peer')

        monkeypatch.setattr(urllib.request, 'urlopen', refuse)
        with pxt_raises(excs.ErrorCode.PROVIDER_ERROR, match='connection reset by peer'):
            db_update(_DB_URI)
        assert len(attempts) == 2
        assert control_plane.submissions == []

    def test_no_wait(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        control_plane = _ControlPlane(generation=7, settled=_receipt(outcome=ReceiptOutcome.OBSERVED))
        _serve(monkeypatch, control_plane)

        accepted = db_update(_DB_URI, wait=False)
        assert accepted.status == 'accepted'
        assert accepted.resolution == 'update_additive', 'nothing has taken effect yet'
        assert control_plane.receipt_reads == 0

    @pytest.mark.parametrize(
        ('fields', 'reason'),
        [
            ({'error': ReceiptError(message='the image build failed', retryable=False)}, 'the image build failed'),
            ({'phase': ResourcePhase.FAILED}, 'no reason was reported'),
        ],
        ids=['error', 'phase'],
    )
    def test_failed(self, project: Path, monkeypatch: pytest.MonkeyPatch, fields: dict, reason: str) -> None:
        _serve(monkeypatch, _ControlPlane(generation=7, settled=_receipt(**fields)))

        with pxt_raises(excs.ErrorCode.PROVIDER_ERROR, match=rf'{reason}\nRun `pxt db retry'):
            db_update(_DB_URI)

    def test_failed_generation_is_not_agreement(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed current generation that declares the project is reported, and update submits it again."""
        failed = _receipt(error=ReceiptError(message='the image build failed', retryable=False))
        control_plane = _ControlPlane(
            generation=8, settled=_receipt(outcome=ReceiptOutcome.OBSERVED), in_agreement=True, current=failed
        )
        _serve(monkeypatch, control_plane)

        plan = db_diff(_DB_URI)
        assert not plan.in_agreement
        assert [op.target for op in plan.ops] == ['generation']
        assert 'failed: the image build failed' in plan.ops[0].description
        assert plan.receipts == [failed]

        db_update(_DB_URI)
        assert len(control_plane.submissions) == 1

    def test_refused(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A refused submission says which generation the database has moved to since it was planned."""
        control_plane = _ControlPlane(
            generation=7, settled=_receipt(outcome=ReceiptOutcome.OBSERVED), drift=1, refuse=True
        )
        _serve(monkeypatch, control_plane)

        with pxt_raises(
            excs.ErrorCode.CONCURRENT_MODIFICATION,
            match=r'pxt://acme:main is at generation 9, not 7\nRun the command again to plan against them',
        ):
            db_update(_DB_URI)

    def test_failure_ends_the_wait(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed receipt is reported at once, while another one is still pending."""
        failed = _receipt(error=ReceiptError(message='the image build failed', retryable=False))
        pending = _receipt(phase=ResourcePhase.BLOCKED).model_copy(update={'resource_id': 'svc-uuid'})
        monkeypatch.setattr(management_client, 'api_call', lambda *a, **k: pytest.fail('the wait polled'))

        with pxt_raises(excs.ErrorCode.PROVIDER_ERROR, match='the image build failed'):
            receipts.await_receipts(PxtPath.parse(_DB_URI, allow_empty_path=True), [pending, failed])

    def test_unknown_outcome(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An outcome this version does not know is not taken for success."""
        _serve(monkeypatch, _ControlPlane(generation=7, settled=_receipt(outcome='ROLLED_BACK')))

        with pxt_raises(excs.ErrorCode.INTERNAL_ERROR, match='ended as ROLLED_BACK'):
            db_update(_DB_URI)

    def test_build_image_pins_generation(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rebuild submits against the generation it read before packaging, not one read afterwards."""
        control_plane = _ControlPlane(generation=7, settled=_receipt(outcome=ReceiptOutcome.OBSERVED), drift=1)
        _serve(monkeypatch, control_plane)

        _, accepted = db_build_image(_DB_URI)
        assert [r.observed for r in accepted] == [True]
        (submitted,) = control_plane.submissions
        assert submitted.expected_generations.database == 7
        assert submitted.database_target is not None
        assert submitted.database_target.cpu == 4.0, 'the desired capacity, so that a resize under way is not undone'
        assert submitted.database_target.pxt_md_version == metadata.VERSION

    def test_superseded(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve(monkeypatch, _ControlPlane(generation=7, settled=_receipt(outcome=ReceiptOutcome.SUPERSEDED)))

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match='replaced by a newer update'):
            db_update(_DB_URI)


class TestResend:
    """A request whose response is lost is resent only when a second delivery changes nothing."""

    @pytest.fixture
    def flaky(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        sent: list[str] = []

        def post(url: str, data: str, **kwargs: Any) -> requests.Response:
            sent.append(data)
            if len(sent) == 1:
                raise requests.exceptions.ConnectionError('connection reset by peer')
            response = requests.Response()
            response.status_code = 200
            response._content = b'{}'
            return response

        monkeypatch.setattr(management_client.SESSION, 'post', post)
        monkeypatch.setattr(management_client, '_IDEMPOTENT_BACKOFF', 0.0)
        return sent

    def test_submission(self, flaky: list[str]) -> None:
        submission = SubmitUpdateRequest(
            db='main', database_target=DatabaseTarget(), expected_generations=ExpectedGenerations(database=1)
        )
        management_client.api_call(submission, management_client.Credential('api_key', 'key', 'test'))
        assert len(flaky) == 2 and flaky[0] == flaky[1], 'the same submission, sent twice'

    def test_secret(self, flaky: list[str]) -> None:
        credential = management_client.Credential('api_key', 'key', 'test')
        with pytest.raises(requests.exceptions.ConnectionError):
            management_client.api_call(SetSecretRequest(org='acme', key='K', value='v'), credential)
        assert len(flaky) == 1
