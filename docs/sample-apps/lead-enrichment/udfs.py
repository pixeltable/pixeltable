"""Importable UDFs for the lead enrichment and outreach pipeline."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import io
import json
import os
import re
import time
import uuid
import wave
from typing import Any

import httpx

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.functions.openai import chat_completions
from pixeltable.utils.local_store import TempStore

_TREG_NAMESPACE = uuid.UUID('6ba7b811-9dad-11d1-80b4-00c04fd430c8')

# Retries run through this pool (default 600 RPM): a retry repeats the deterministic Idempotency-Key,
# so treg replays or coalesces the call instead of charging twice.
_TREG_POOL = 'request-rate:treg'


def _check_response(response: httpx.Response) -> None:
    status = response.status_code
    if status >= 400:
        code = {
            401: excs.ErrorCode.PROVIDER_AUTH_ERROR,
            403: excs.ErrorCode.PROVIDER_AUTH_ERROR,
            409: excs.ErrorCode.PROVIDER_OVERLOADED,
            429: excs.ErrorCode.RATE_LIMITED,
            503: excs.ErrorCode.PROVIDER_OVERLOADED,
        }.get(status, excs.ErrorCode.PROVIDER_ERROR if status >= 500 else excs.ErrorCode.PROVIDER_BAD_REQUEST)
        error = excs.ExternalServiceError(code, f'treg returned HTTP {status}', provider='treg', status_code=status)
        # The request-rate scheduler retries only when the error's httpx response carries a retryable status,
        # not by ErrorCode.is_retryable. A 409 is an in-flight idempotency key; treat it like a busy provider.
        error.response = httpx.Response(503, headers=response.headers) if status == 409 or status >= 500 else response  # type: ignore[attr-defined]
        raise error
    if response.headers.get('X-Treg-Upstream-Failed'):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_BAD_REQUEST, 'treg upstream task failed', provider='treg'
        )


def _cost_micro(response: httpx.Response) -> int:
    # treg omits the cost header when the team's own key served the call, and that call is not billed.
    header = (
        'X-Treg-Original-Cost-Micro'
        if response.headers.get('X-Treg-Idempotent-Replay') == 'true'
        else 'X-Treg-Cost-Micro'
    )
    return int(response.headers.get(header, 0))


@dataclasses.dataclass
class _TregResult:
    """A settled treg call: the routed output or the binary body, plus its billing receipt."""

    output: Any  # the 'output' field of a routed JSON endpoint
    content: bytes  # the raw body of a binary endpoint
    content_type: str
    cost_micro: int
    call_id: str | None
    served_by: str | None
    verification: dict[str, Any] | None


def _at_path(document: Any, path: str) -> Any:
    for part in path.split('.'):
        document = document.get(part) if isinstance(document, dict) else None
    return document


async def _poll_task(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    *,
    served_by: str,
    call_ref: str,
    task_id: str,
    prior_cost_micro: int,
) -> tuple[Any, int]:
    """Poll the submitted child, never resubmit a routed operation with a new key."""
    catalog = await client.get(f'{base_url}/catalog/endpoints/{served_by}')
    _check_response(catalog)
    descriptor = catalog.json()['endpoint']['async']
    poll = descriptor['poll']
    endpoint = poll['endpoint']
    # Only catalog ids go through /call/. Never forward the token to a provider URL.
    if '/' in endpoint or ':' in endpoint:
        raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'Unsupported treg poll endpoint')
    deadline = time.monotonic() + min(float(descriptor.get('max_age', 300)), 300)
    status_rule = descriptor['status']
    while time.monotonic() < deadline:
        await asyncio.sleep(float(descriptor.get('interval', 30)))
        result = await client.get(
            f'{base_url}/call/{endpoint}', headers=headers, params={poll['param']['name']: task_id}
        )
        _check_response(result)
        document = result.json()
        status = _at_path(document, status_rule['path'])
        if status in status_rule.get('success', []):
            # Polls return provider-native JSON, not the routed {output, _treg} envelope.
            # Only accept the documented canonical fields; unsupported shapes fail in the UDF.
            output = (
                _at_path(document, descriptor.get('result', {}).get('path', ''))
                if descriptor.get('result', {}).get('path')
                else document
            )
            receipt = await client.get(f'{base_url}/calls/{call_ref}', headers=headers)
            _check_response(receipt)
            return output, prior_cost_micro + int(receipt.json()['charged_micro'])
        if status in status_rule.get('failure', []) + status_rule.get('billed_failure', []):
            raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_BAD_REQUEST, f'treg task {call_ref} failed')
    raise excs.ExternalServiceError(
        excs.ErrorCode.PROVIDER_TIMEOUT,
        f'treg task {call_ref} is still pending. Inspect this call before starting new work.',
    )


async def _finish_verification(
    client: httpx.AsyncClient, base_url: str, headers: dict[str, str], result: _TregResult
) -> None:
    """Settle the asynchronous verifier the routed call started, into `result`."""
    verification = result.verification
    assert verification is not None
    if verification['served_by'] != 'bounceban.people.email.verify':
        raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'Unsupported pending verifier schema')
    call = await client.get(f'{base_url}/calls/{verification["call_id"]}', headers=headers)
    _check_response(call)
    task = call.json().get('async_task')
    if task is None:
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR, f'No pollable task for {verification["call_id"]}'
        )
    output, charged_micro = await _poll_task(
        client,
        base_url,
        headers,
        served_by=verification['served_by'],
        call_ref=verification['call_id'],
        task_id=task['task_id'],
        prior_cost_micro=0,
    )
    # A null result is a billed miss. Keep the email that the find call already returned, marked unknown.
    verdicts = {'deliverable': 'valid', 'undeliverable': 'invalid', 'risky': 'risky', 'unknown': 'unknown'}
    verdict = 'unknown' if output is None else verdicts.get(output)
    if verdict is None:
        raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'Unsupported BounceBan verdict')
    verification.update(pending=False, verdict=verdict, cost_micro=charged_micro)
    result.cost_micro += charged_micro


async def _settle_response(
    client: httpx.AsyncClient, base_url: str, headers: dict[str, str], response: httpx.Response, *, verify: bool
) -> _TregResult:
    """Resolve a routed call through pending tasks and verification into a settled result."""
    _check_response(response)
    data = response.json() if response.headers.get('Content-Type', '').startswith('application/json') else {}
    meta = data.get('_treg', {})
    pending = response.status_code == 202 or meta.get('outcome') == 'pending'
    result = _TregResult(
        output=data.get('output'),
        content=response.content,
        content_type=response.headers.get('Content-Type', ''),
        # A pending 202 carries reserved_micro, not a charge. Its cost is the parent's settled charge (null until
        # settled) plus the child receipt that _poll_task adds.
        cost_micro=0 if pending else _cost_micro(response),
        call_id=response.headers.get('X-Treg-Call-Id') or meta.get('call_ref'),
        served_by=response.headers.get('X-Treg-Served-By') or meta.get('served_by'),
        verification=meta.get('verification'),
    )
    if pending:
        # A pending call settles through its catalog descriptor, as a JSON document.
        result.output, result.cost_micro = await _poll_task(
            client,
            base_url,
            headers,
            served_by=meta['served_by'],
            call_ref=meta['call_ref'],
            task_id=meta['async']['task_id'],
            prior_cost_micro=meta.get('charged_micro') or 0,
        )
        result.call_id = meta['call_ref']
        result.served_by = meta['served_by']
        result.content = b''
        result.content_type = 'application/json'
    if verify and (result.verification or {}).get('pending'):
        await _finish_verification(client, base_url, headers, result)
    return result


async def _treg_post(
    endpoint: str,
    body: dict[str, Any],
    *,
    lead_id: uuid.UUID,
    max_cost_usd: float,
    verify: bool = False,
    timeout_s: float = 90.0,
    extra_headers: dict[str, str] | None = None,
) -> _TregResult:
    token = os.environ.get('TREG_TOKEN', '')
    if not token:
        raise excs.AuthorizationError(
            excs.ErrorCode.MISSING_CREDENTIALS, 'Export TREG_TOKEN before starting Pixeltable.'
        )
    base_url = os.environ.get('TREG_BASE_URL', 'https://treg.to').rstrip('/')
    request_options = {
        'endpoint': endpoint,
        'body': body,
        'max_cost_usd': max_cost_usd,
        'verify': verify,
        'headers': extra_headers,
    }
    seed = f'{lead_id}:{json.dumps(request_options, sort_keys=True, separators=(",", ":"))}'
    headers = {
        'X-Treg-Token': token,
        'Idempotency-Key': str(uuid.uuid5(_TREG_NAMESPACE, seed)),
        'X-Treg-Route-Max-Cost': str(max_cost_usd),
        **(extra_headers or {}),
    }
    if verify:
        headers['X-Treg-Route-Verify'] = 'true'
    if org := os.environ.get('TREG_ORG'):
        headers['X-Treg-Org'] = org
    auth_headers = {key: value for key, value in headers.items() if key in ('X-Treg-Token', 'X-Treg-Org')}
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(f'{base_url}/call/{endpoint}', headers=headers, json=body)
            return await _settle_response(client, base_url, auth_headers, response, verify=verify)
    except httpx.TransportError as cause:
        timeout = isinstance(cause, httpx.TimeoutException)
        error = excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_TIMEOUT if timeout else excs.ErrorCode.PROVIDER_ERROR,
            f'treg {"timed out" if timeout else "connection error"} on {endpoint}',
            provider='treg',
        )
        error.response = httpx.Response(504 if timeout else 503)  # type: ignore[attr-defined]
        raise error from cause


@pxt.udf(is_deterministic=False, resource_pool=_TREG_POOL)
async def treg_find_email(
    lead_id: uuid.UUID, full_name: str, domain: str, max_cost_usd: float = 0.03
) -> dict[str, Any]:
    """Find a work email and request mailbox verification within the spend ceiling."""
    if os.environ.get('LEAD_ENRICHMENT_DEMO') == '1':
        return {
            'email': None,
            'verified': False,
            'verdict': 'unknown',
            'served_by': 'demo fixture',
            'call_id': f'demo:{lead_id}',
            'cost_micro': 0,
        }
    result = await _treg_post(
        'treg.people.email.find',
        {'full_name': full_name, 'domain': domain},
        lead_id=lead_id,
        max_cost_usd=max_cost_usd,
        verify=True,
    )
    output = result.output
    if not isinstance(output, dict) or 'email' not in output:
        raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'Unsupported treg email result schema')
    verification = result.verification or {}
    return {
        'email': output['email'],
        'verified': bool(output.get('verified')),
        'verdict': verification.get('verdict'),
        'served_by': result.served_by,
        'call_id': result.call_id,
        'cost_micro': result.cost_micro,
    }


@pxt.udf(is_deterministic=False, resource_pool=_TREG_POOL)
async def treg_extract_site(lead_id: uuid.UUID, domain: str, max_cost_usd: float = 0.02) -> dict[str, Any]:
    """Read text from the provider-native page objects in output.pages."""
    if os.environ.get('LEAD_ENRICHMENT_DEMO') == '1':
        return {
            'url': domain,
            'markdown': 'Demo company context. This is synthetic data, not a website extract.',
            'served_by': 'demo fixture',
            'call_id': f'demo:{lead_id}',
            'cost_micro': 0,
        }
    url = domain if domain.startswith(('http://', 'https://')) else f'https://{domain}'
    result = await _treg_post('treg.web.extract', {'url': url}, lead_id=lead_id, max_cost_usd=max_cost_usd)
    output = result.output
    pages = output.get('pages') if isinstance(output, dict) else None
    if not isinstance(pages, list) or not all(isinstance(page, dict) for page in pages):
        raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'Unsupported treg website result schema')
    excerpts = []
    for page in pages:
        text = page.get('markdown') or page.get('text') or page.get('raw_content') or page.get('excerpts') or ''
        if isinstance(text, list):
            text = '\n'.join(text)
        excerpts.append(text)
    return {
        'url': url,
        'markdown': '\n\n'.join(excerpts)[:4000],
        'served_by': result.served_by,
        'call_id': result.call_id,
        'cost_micro': result.cost_micro,
    }


@pxt.udf(is_deterministic=False)
async def outreach_copywriter(prompt: str, model: str = 'gpt-4o-mini') -> dict[str, Any]:
    """Return explicit demo copy or call the built-in OpenAI integration; propagate provider failures."""
    if os.environ.get('LEAD_ENRICHMENT_DEMO') == '1':
        return {'choices': [{'message': {'content': 'Demo outreach draft. Replace with live enrichment before use.'}}]}
    return await chat_completions.py_fn(model=model, messages=[{'role': 'user', 'content': prompt}])


@pxt.udf(is_deterministic=False, resource_pool=_TREG_POOL)
async def treg_voice_note(lead_id: uuid.UUID, script: str, max_cost_usd: float = 0.02) -> dict[str, Any]:
    """Keep the audio and billing receipt together so computing cost never repeats synthesis."""
    if os.environ.get('LEAD_ENRICHMENT_DEMO') == '1':
        audio = io.BytesIO()
        with wave.open(audio, 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(bytes(16000))
        return {
            'audio_base64': base64.b64encode(audio.getvalue()).decode('ascii'),
            'format': 'wav',
            'call_id': f'demo:{lead_id}',
            'cost_micro': 0,
        }
    result = await _treg_post(
        'fishaudio.tts.s2-1-pro',
        {'text': script, 'format': 'mp3'},
        lead_id=lead_id,
        max_cost_usd=max_cost_usd,
        extra_headers={'model': 's2.1-pro'},
    )
    # A pending or errored call settles as a JSON document; storing it would corrupt the audio column.
    if not result.content_type.startswith('audio/'):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'treg voice synthesis returned {result.content_type or "no Content-Type"}, not audio',
        )
    return {
        'audio_base64': base64.b64encode(result.content).decode('ascii'),
        'call_id': result.call_id,
        'cost_micro': result.cost_micro,
    }


@pxt.udf
def voice_note_audio(receipt: dict[str, Any]) -> pxt.Audio:
    """Persist the audio from the stored receipt, including on later recomputations."""
    output_path = TempStore.create_path(extension=f'.{receipt.get("format", "mp3")}')
    output_path.write_bytes(base64.b64decode(receipt['audio_base64']))
    return str(output_path)


@pxt.udf
def outreach_prompt(full_name: str, role: str | None, domain: str, site_markdown: str) -> str:
    """Build the cold-email prompt; a null role drops the parenthetical instead of formatting 'None'."""
    recipient = f'{full_name} ({role})' if role else full_name
    return (
        f'Write a concise 3-sentence cold outreach email to {recipient} at {domain}.\n'
        'Ground the opener in the website excerpt. If it is empty, do not invent company facts. No placeholders.\n\n'
        f'WEBSITE EXCERPT:\n{site_markdown}'
    )


@pxt.udf
def lead_tier(verdict: str | None, role: str | None) -> str:
    """Classify outreach readiness from mailbox verification and role."""
    words = set(re.findall(r'[a-z]+', (role or '').lower()))
    is_exec = bool(words & {'founder', 'cofounder', 'ceo', 'cto', 'vp', 'head', 'director', 'chief'})
    if (verdict or '').lower() == 'valid':
        return 'priority' if is_exec else 'ready'
    return 'review' if (verdict or '').lower() in ('catch_all', 'risky', 'unknown') else 'skip'
