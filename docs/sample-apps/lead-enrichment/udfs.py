"""UDFs for the lead enrichment and outreach pipeline (pure Python, no table imports).

Pixeltable records UDFs by module path (`udfs.treg_find_email`, etc.), so they live in
their own importable module rather than inline in `app.py`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from typing import Any

import httpx

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.utils.local_store import TempStore

_TREG_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


async def _treg_post(
    endpoint: str,
    body: dict[str, Any],
    *,
    idempotency_seed: str,
    max_cost_usd: float,
    verify: bool = False,
    timeout_s: float = 90.0,
) -> httpx.Response:
    """Single call site for treg.to requests with idempotency, spend ceiling, and Pixeltable error mapping."""
    token = os.environ.get("TREG_TOKEN", "")
    if not token:
        raise excs.AuthorizationError(
            excs.ErrorCode.MISSING_CREDENTIALS,
            "TREG_TOKEN is not set. Export TREG_TOKEN before starting the Pixeltable daemon.",
        )
    base_url = os.environ.get("TREG_BASE_URL", "https://treg.to").rstrip("/")
    headers = {
        "X-Treg-Token": token,
        "Idempotency-Key": str(uuid.uuid5(_TREG_NAMESPACE, idempotency_seed)),
        "X-Treg-Route-Max-Cost": str(max_cost_usd),
    }
    if verify:
        headers["X-Treg-Route-Verify"] = "true"
    if org := os.environ.get("TREG_ORG"):
        headers["X-Treg-Org"] = org

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(f"{base_url}/call/{endpoint}", headers=headers, json=body)

    if r.status_code >= 400:
        detail = r.text[:300]
        retry_hdr = r.headers.get("Retry-After")
        retry_after = float(retry_hdr) if retry_hdr and retry_hdr.isdigit() else None
        if r.status_code in (401, 403):
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_AUTH_ERROR, f"treg {r.status_code}: {detail}"
            )
        if r.status_code == 429:
            raise excs.ExternalServiceError(
                excs.ErrorCode.RATE_LIMITED,
                f"treg 429: {detail}",
                retry_after=retry_after,
            )
        if r.status_code == 503:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_OVERLOADED,
                f"treg 503: {detail}",
                retry_after=retry_after,
            )
        if r.status_code >= 500:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f"treg {r.status_code}: {detail}",
                retry_after=retry_after,
            )
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_BAD_REQUEST, f"treg {r.status_code}: {detail}"
        )

    if upstream_failed := r.headers.get("X-Treg-Upstream-Failed"):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_BAD_REQUEST,
            f"treg upstream task failed ({upstream_failed}) on {endpoint}",
        )
    return r


@pxt.udf(is_deterministic=False)
async def treg_find_email(
    full_name: str, domain: str, max_cost_usd: float = 0.03
) -> dict[str, Any]:
    """Find a work email across treg's provider waterfall and verify deliverability in one call."""
    if not os.environ.get("TREG_TOKEN"):
        first = full_name.split()[0].lower() if full_name else "lead"
        last = full_name.split()[-1].lower() if len(full_name.split()) > 1 else ""
        user = f"{first}.{last}" if last and last != first else first
        return {
            "email": f"{user}@{domain}",
            "verified": True,
            "verdict": "valid",
            "served_by": "bounceban (demo)",
            "call_id": f"demo-{uuid.uuid4().hex[:8]}",
            "cost_micro": 12000,
        }

    r = await _treg_post(
        "treg.people.email.find",
        {"full_name": full_name, "domain": domain},
        idempotency_seed=f"email.find:{full_name}:{domain}",
        max_cost_usd=max_cost_usd,
        verify=True,
    )
    data: dict[str, Any] = r.json()
    output = data.get("output") or {}
    treg_meta = data.get("_treg") or {}
    verification = treg_meta.get("verification") or {}
    return {
        "email": output.get("email"),
        "verified": bool(output.get("verified")),
        "verdict": verification.get("verdict"),
        "served_by": r.headers.get("X-Treg-Served-By") or treg_meta.get("served_by"),
        "call_id": r.headers.get("X-Treg-Call-Id"),
        "cost_micro": int(
            r.headers.get("X-Treg-Cost-Micro", treg_meta.get("charged_micro") or 0)
        ),
    }


@pxt.udf(is_deterministic=False)
async def treg_extract_site(domain: str, max_cost_usd: float = 0.02) -> dict[str, Any]:
    """Extract live company website markdown via treg's routed web extractor."""
    url = domain if domain.startswith(("http://", "https://")) else f"https://{domain}"
    if not os.environ.get("TREG_TOKEN"):
        brand = domain.split(".")[0].capitalize()
        return {
            "url": url,
            "markdown": (
                f"# {brand}\n\n"
                f"{brand} builds multimodal AI data infrastructure and declarative application runtimes. "
                "Capabilities include automatic incremental compute graphs, media storage, and real-time query serving."
            ),
            "served_by": "treg.web.extract (demo)",
            "call_id": f"demo-{uuid.uuid4().hex[:8]}",
            "cost_micro": 8000,
        }

    r = await _treg_post(
        "treg.web.extract",
        {"url": url},
        idempotency_seed=f"web.extract:{url}",
        max_cost_usd=max_cost_usd,
        timeout_s=60.0,
    )
    data: dict[str, Any] = r.json()
    output = data.get("output") or data
    markdown = (output.get("markdown") or output.get("text") or "")[:4000]
    return {
        "url": url,
        "markdown": markdown,
        "served_by": r.headers.get("X-Treg-Served-By"),
        "call_id": r.headers.get("X-Treg-Call-Id"),
        "cost_micro": int(r.headers.get("X-Treg-Cost-Micro", 0)),
    }


@pxt.udf(is_deterministic=False)
async def outreach_copywriter(
    full_name: str, role: str | None, domain: str, site_markdown: str | None
) -> str | None:
    """Generate a 3-sentence personalized cold email via OpenAI (or demo copywriter if OPENAI_API_KEY is unset)."""
    if not full_name:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    prompt = (
        f"Write a concise 3-sentence cold outreach email to {full_name} ({role or 'Executive'}) at {domain}.\n"
        f"Ground the opener in something specific from their website excerpt below. No fluff or placeholders.\n\n"
        f"WEBSITE EXCERPT:\n{site_markdown or domain}"
    )
    if api_key:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key)
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    company = domain.split(".")[0].capitalize()
    return (
        f"Hi {full_name.split()[0]},\n\n"
        f"Saw what {company} is building in the AI infrastructure space — the focus on real-time execution caught my eye. "
        f"Teams scaling multimodal apps usually run into painful data synchronization delays and unpredictably high per-call API costs. "
        f"Would you be open to a 10-minute chat next Tuesday to see how our unified ledger cuts pipeline latency and costs?"
    )


@pxt.udf(is_deterministic=False)
async def treg_voice_note(
    script: str | None, max_cost_usd: float = 0.02
) -> pxt.Audio | None:
    """Synthesize a personalized MP3 voice note via Fish Audio on treg and write it to TempStore."""
    if not script:
        return None
    if not os.environ.get("TREG_TOKEN"):
        output_path = TempStore.create_path(extension=".mp3")
        try:
            if shutil.which("ffmpeg"):
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=44100:cl=mono",
                        "-t",
                        "1",
                        "-acodec",
                        "libmp3lame",
                        "-b:a",
                        "32k",
                        str(output_path),
                    ],
                    check=True,
                    capture_output=True,
                )
                return str(output_path)
        except Exception:
            pass
        return None

    r = await _treg_post(
        "fishaudio.tts.s2-1-pro",
        {"text": script, "format": "mp3"},
        idempotency_seed=f"tts:{script[:120]}",
        max_cost_usd=max_cost_usd,
        timeout_s=60.0,
    )
    output_path = TempStore.create_path(extension=".mp3")
    output_path.write_bytes(r.content)
    return str(output_path)


@pxt.udf
def lead_tier(verdict: str | None, role: str | None) -> str | None:
    """Classify outreach readiness from mailbox verification verdict and executive role."""
    if not verdict and not role:
        return None
    v = (verdict or "").lower()
    r = (role or "").lower()
    is_exec = any(
        k in r for k in ("founder", "ceo", "cto", "vp", "head", "director", "chief")
    )
    if v == "valid" and is_exec:
        return "priority"
    if v == "valid":
        return "ready"
    if v in ("catch_all", "risky", "unknown"):
        return "review"
    return "skip"
