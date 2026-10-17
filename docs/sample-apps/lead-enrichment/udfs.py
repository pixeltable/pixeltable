"""Importable UDFs for the lead enrichment and outreach pipeline."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
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
        # The scheduler reads the httpx response, not ExternalServiceError.is_retryable.
        # A 409 is an in-flight idempotency key; treat it like a busy provider.
        error.response = httpx.Response(503, headers=response.headers) if status == 409 or status >= 500 else response  # type: ignore[attr-defined]
        raise error
    if response.headers.get('X-Treg-Upstream-Failed'):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_BAD_REQUEST, 'treg upstream task failed', provider='treg'
        )


def _cost_micro(response: httpx.Response) -> int:
    header = (
        'X-Treg-Original-Cost-Micro'
        if response.headers.get('X-Treg-Idempotent-Replay') == 'true'
        else 'X-Treg-Cost-Micro'
    )
    return int(response.headers[header])


def _at_path(document: Any, path: str) -> Any:
    for part in path.split('.'):
        document = document.get(part) if isinstance(document, dict) else None
    return document


async def _poll_pending(
    client: httpx.AsyncClient, base_url: str, headers: dict[str, str], response: httpx.Response
) -> httpx.Response:
    """Poll the submitted child, never resubmit a routed operation with a new key."""
    data = response.json()
    meta = data['_treg']
    pending = meta['async']
    catalog = await client.get(f'{base_url}/catalog/endpoints/{meta["served_by"]}')
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
            f'{base_url}/call/{endpoint}', headers=headers, params={poll['param']['name']: pending['task_id']}
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
            receipt = await client.get(f'{base_url}/calls/{meta["call_ref"]}', headers=headers)
            _check_response(receipt)
            cost = _cost_micro(response) + int(receipt.json()['charged_micro'])
            meta.update(outcome='hit', charged_micro=cost)
            return httpx.Response(
                200,
                json={'output': output, 'raw': document, '_treg': meta},
                headers={
                    'X-Treg-Cost-Micro': str(cost),
                    'X-Treg-Call-Id': meta['call_ref'],
                    'X-Treg-Served-By': meta['served_by'],
                },
            )
        if status in status_rule.get('failure', []) + status_rule.get('billed_failure', []):
            raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_BAD_REQUEST, f'treg task {meta["call_ref"]} failed')
    raise excs.ExternalServiceError(
        excs.ErrorCode.PROVIDER_TIMEOUT,
        f'treg task {meta["call_ref"]} is still pending. Inspect this call before starting new work.',
    )


async def _finish_verification(
    client: httpx.AsyncClient, base_url: str, headers: dict[str, str], response: httpx.Response
) -> httpx.Response:
    data = response.json()
    verification = data['_treg']['verification']
    if verification['served_by'] != 'bounceban.people.email.verify':
        raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'Unsupported pending verifier schema')
    call = await client.get(f'{base_url}/calls/{verification["call_id"]}', headers=headers)
    _check_response(call)
    task = call.json().get('async_task')
    if task is None:
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR, f'No pollable task for {verification["call_id"]}'
        )
    pending = httpx.Response(
        202,
        json={
            '_treg': {
                'served_by': verification['served_by'],
                'call_ref': verification['call_id'],
                'async': {'task_id': task['task_id']},
            }
        },
        headers={'X-Treg-Cost-Micro': '0'},
    )
    completed = await _poll_pending(client, base_url, headers, pending)
    verdict = {'deliverable': 'valid', 'undeliverable': 'invalid', 'risky': 'risky', 'unknown': 'unknown'}.get(
        completed.json()['output']
    )
    if verdict is None:
        raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'Unsupported BounceBan verdict')
    cost = _cost_micro(response) + _cost_micro(completed)
    verification.update(pending=False, verdict=verdict, cost_micro=_cost_micro(completed))
    data['_treg']['charged_micro'] = cost
    result_headers = dict(response.headers)
    # Replace replay headers: this receipt now includes the settled asynchronous verifier.
    result_headers.pop('x-treg-idempotent-replay', None)
    result_headers['x-treg-cost-micro'] = str(cost)
    result_headers.pop('content-length', None)
    result_headers.pop('content-encoding', None)
    return httpx.Response(200, json=data, headers=result_headers)


async def _treg_post(
    endpoint: str,
    body: dict[str, Any],
    *,
    lead_id: uuid.UUID,
    max_cost_usd: float,
    verify: bool = False,
    timeout_s: float = 90.0,
    extra_headers: dict[str, str] | None = None,
) -> httpx.Response:
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
            _check_response(response)
            if response.status_code == 202 or (
                response.headers.get('Content-Type', '').startswith('application/json')
                and response.json().get('_treg', {}).get('outcome') == 'pending'
            ):
                response = await _poll_pending(client, base_url, auth_headers, response)
            if verify and response.json().get('_treg', {}).get('verification', {}).get('pending'):
                response = await _finish_verification(client, base_url, auth_headers, response)
            return response
    except httpx.TransportError as cause:
        timeout = isinstance(cause, httpx.TimeoutException)
        error = excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_TIMEOUT if timeout else excs.ErrorCode.PROVIDER_ERROR,
            f'treg {"timed out" if timeout else "connection error"} on {endpoint}',
            provider='treg',
        )
        error.response = httpx.Response(504 if timeout else 503)  # type: ignore[attr-defined]
        raise error from cause


@pxt.udf(is_deterministic=False)
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
    response = await _treg_post(
        'treg.people.email.find',
        {'full_name': full_name, 'domain': domain},
        lead_id=lead_id,
        max_cost_usd=max_cost_usd,
        verify=True,
    )
    data = response.json()
    output = data['output']
    if not isinstance(output, dict) or 'email' not in output:
        raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'Unsupported treg email result schema')
    meta = data['_treg']
    verification = meta.get('verification', {})
    # A pending verifier is not evidence of a deliverable mailbox.
    if verification.get('pending'):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_TIMEOUT,
            f'Mailbox verification is still pending: {verification["call_id"]}. Inspect the call before retrying.',
        )
    return {
        'email': output['email'],
        'verified': bool(output.get('verified')),
        'verdict': verification.get('verdict'),
        'served_by': response.headers.get('X-Treg-Served-By') or meta.get('served_by'),
        'call_id': response.headers.get('X-Treg-Call-Id'),
        'cost_micro': _cost_micro(response),
    }


@pxt.udf(is_deterministic=False)
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
    response = await _treg_post('treg.web.extract', {'url': url}, lead_id=lead_id, max_cost_usd=max_cost_usd)
    output = response.json()['output']
    if not isinstance(output, dict) or not isinstance(output.get('pages'), list):
        raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'Unsupported treg website result schema')
    pages = output['pages']
    excerpts = []
    for page in pages:
        text = page.get('markdown') or page.get('text') or page.get('raw_content') or page.get('excerpts') or ''
        if isinstance(text, list):
            text = '\n'.join(text)
        excerpts.append(text)
    return {
        'url': url,
        'markdown': '\n\n'.join(excerpts)[:4000],
        'served_by': response.headers.get('X-Treg-Served-By'),
        'call_id': response.headers.get('X-Treg-Call-Id'),
        'cost_micro': _cost_micro(response),
    }


@pxt.udf(is_deterministic=False)
async def outreach_copywriter(prompt: str, model: str = 'gpt-4o-mini') -> dict[str, Any]:
    """Return explicit demo copy or call the built-in OpenAI integration; propagate provider failures."""
    if os.environ.get('LEAD_ENRICHMENT_DEMO') == '1':
        return {'choices': [{'message': {'content': 'Demo outreach draft. Replace with live enrichment before use.'}}]}
    return await chat_completions.py_fn(model=model, messages=[{'role': 'user', 'content': prompt}])


@pxt.udf(is_deterministic=False)
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
    response = await _treg_post(
        'fishaudio.tts.s2-1-pro',
        {'text': script, 'format': 'mp3'},
        lead_id=lead_id,
        max_cost_usd=max_cost_usd,
        extra_headers={'model': 's2.1-pro'},
    )
    return {
        'audio_base64': base64.b64encode(response.content).decode('ascii'),
        'call_id': response.headers.get('X-Treg-Call-Id'),
        'cost_micro': _cost_micro(response),
    }


@pxt.udf
def voice_note_audio(receipt: dict[str, Any]) -> pxt.Audio:
    """Persist the audio from the stored receipt, including on later recomputations."""
    output_path = TempStore.create_path(extension=f'.{receipt.get("format", "mp3")}')
    output_path.write_bytes(base64.b64decode(receipt['audio_base64']))
    return str(output_path)


@pxt.udf
def lead_tier(verdict: str | None, role: str | None) -> str:
    """Classify outreach readiness from mailbox verification and role."""
    is_exec = any(k in (role or '').lower() for k in ('founder', 'ceo', 'cto', 'vp', 'head', 'director', 'chief'))
    if (verdict or '').lower() == 'valid':
        return 'priority' if is_exec else 'ready'
    return 'review' if (verdict or '').lower() in ('catch_all', 'risky', 'unknown') else 'skip'
