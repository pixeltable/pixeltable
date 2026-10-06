"""Submitting changes to a hosted database or its services, and waiting on the resulting receipts.

Waiting is read-only: the control plane completes an accepted generation whether or not anyone waits.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from pixeltable import catalog, exceptions as excs
from pixeltable.service import management_client
from pixeltable.service.management_protocol import (
    ExpectedGenerations,
    GetReceiptsRequest,
    GetReceiptsResponse,
    PrepareUpdateRequest,
    PrepareUpdateResponse,
    ReceiptRef,
    RetryRequest,
    RetryResponse,
    SubmitUpdateRequest,
    SubmitUpdateResponse,
)
from pixeltable_cli.types import GenerationReceipt

POLL_INTERVAL = 5.0
TIMEOUT = 3 * 3600.0


def submit(db_path: catalog.Path, request: SubmitUpdateRequest) -> list[GenerationReceipt]:
    """Submit request and return the resulting receipts.

    On a refusal, re-read the current generations and raise with the ones that moved.
    """
    try:
        response = management_client.api_call(request)
    except excs.ConcurrencyError as exc:
        prepared = PrepareUpdateResponse.model_validate(
            management_client.api_call(
                PrepareUpdateRequest(
                    org=request.org,
                    db=request.db,
                    database_target=request.database_target,
                    service_mutations=request.service_mutations,
                )
            )
        )
        moved = _moved_generations(db_path, request, prepared.generations)
        if len(moved) == 0:
            raise
        message = '\n'.join([exc.message, *moved, 'Run the command again to plan against them.'])
        raise excs.ConcurrencyError(excs.ErrorCode.CONCURRENT_MODIFICATION, message) from exc
    return SubmitUpdateResponse.model_validate(response).receipts


def _moved_generations(db_path: catalog.Path, request: SubmitUpdateRequest, current: ExpectedGenerations) -> list[str]:
    """One line per resource whose generation changed since request was prepared."""
    expected = request.expected_generations
    lines: list[str] = []
    if request.database_target is not None and current.database != expected.database:
        lines.append(f'{db_path.uri_str} is at generation {current.database}, not {expected.database}')
    for mutation in request.service_mutations:
        was = expected.service(mutation.service_name, mutation.base_path)
        now = current.service(mutation.service_name, mutation.base_path)
        if now != was:
            uri = '/'.join(part for part in (db_path.uri_str, mutation.base_path, mutation.service_name) if part)
            lines.append(f'service {uri} is at generation {now}, not {was}')
    return lines


def await_receipts(db_path: catalog.Path, receipts: Sequence[GenerationReceipt]) -> list[GenerationReceipt]:
    """Poll receipts until each is settled, and return them.

    Raises if one failed, was superseded, or is still pending at the deadline.
    """
    if len(receipts) == 0:
        raise excs.InternalError(
            excs.ErrorCode.INTERNAL_ERROR, f'{db_path.uri_str} accepted a change without a receipt'
        )
    current = list(receipts)
    deadline = time.monotonic() + TIMEOUT
    while not all(r.settled for r in current):
        raise_if_unsuccessful(db_path, current)
        if time.monotonic() >= deadline:
            pending = next(r for r in current if not r.settled)
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_TIMEOUT,
                f'{_describe(db_path, pending)} is still {pending.progress} after {int(TIMEOUT)}s. '
                f'The change continues in Pixeltable Cloud; run {_status_hint(db_path, pending)} to check its '
                'progress.',
                provider='pixeltable_cloud',
            )
        time.sleep(POLL_INTERVAL)
        current = read_receipts(db_path, current)
    raise_if_unsuccessful(db_path, current)
    return current


def read_receipts(db_path: catalog.Path, receipts: Sequence[GenerationReceipt]) -> list[GenerationReceipt]:
    response = management_client.api_call(
        GetReceiptsRequest(org=db_path.org, receipts=[ReceiptRef.of(r) for r in receipts])
    )
    read = GetReceiptsResponse.model_validate(response).receipts
    if [ReceiptRef.of(r) for r in read] != [ReceiptRef.of(r) for r in receipts]:
        raise excs.InternalError(
            excs.ErrorCode.INTERNAL_ERROR, f'{db_path.uri_str} returned receipts that do not match the request'
        )
    return read


def raise_if_unsuccessful(db_path: catalog.Path, receipts: Sequence[GenerationReceipt]) -> None:
    """Raise for the first receipt that failed or was superseded."""
    for r in receipts:
        if r.failed:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'{_describe(db_path, r)} failed: {r.failure_reason}\n{_retry_hint(db_path, r)}',
                provider='pixeltable_cloud',
            )
        if r.superseded:
            raise excs.ConcurrencyError(
                excs.ErrorCode.CONCURRENT_MODIFICATION,
                f'{_describe(db_path, r)} was replaced by a newer update before it finished',
            )
        if r.outcome is not None and not r.observed:
            raise excs.InternalError(
                excs.ErrorCode.INTERNAL_ERROR,
                f'{_describe(db_path, r)} ended with unknown outcome {r.outcome}; upgrade Pixeltable to read it',
            )


def retry(db_path: catalog.Path, receipt: GenerationReceipt) -> GenerationReceipt:
    """Start a new attempt of receipt's generation, which must be current and failed."""
    request = RetryRequest(
        org=db_path.org,
        db=_db(db_path),
        kind='service' if receipt.service_name is not None else 'database',
        service_name=receipt.service_name,
        base_path=receipt.base_path,
        generation=receipt.generation,
    )
    return RetryResponse.model_validate(management_client.api_call(request)).receipt


def _db(db_path: catalog.Path) -> str:
    assert db_path.db is not None
    return db_path.db


def _describe(db_path: catalog.Path, receipt: GenerationReceipt) -> str:
    if receipt.service_name is None:
        return f'{db_path.uri_str} generation {receipt.generation}'
    return f'service {_service_uri(db_path, receipt)} generation {receipt.generation}'


def _service_uri(db_path: catalog.Path, receipt: GenerationReceipt) -> str:
    return '/'.join(part for part in (db_path.uri_str, receipt.base_path, receipt.service_name) if part)


def _status_hint(db_path: catalog.Path, receipt: GenerationReceipt) -> str:
    if receipt.service_name is None:
        return f'`pxt db status {db_path.uri_str}`'
    return f'`pxt service list {db_path.uri_str}`'


def _retry_hint(db_path: catalog.Path, receipt: GenerationReceipt) -> str:
    if receipt.service_name is None:
        return f'Run `pxt db retry {db_path.uri_str}` to try it again, or change the project and update again.'
    return (
        f'Run `pxt service retry {_service_uri(db_path, receipt)}` to try it again, or change the service and '
        'update again.'
    )
