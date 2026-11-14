"""Lead Enrichment & Multimodal Outreach API built with Pixeltable and treg.to.

Insert a prospect (lead_id, full_name, domain, role, stage). Pixeltable runs a declarative
enrichment DAG: waterfall email discovery + BounceBan mailbox verification via treg.to,
live website extraction via treg.to, a personalized 3-sentence cold email via OpenAI,
a personalized Fish Audio MP3 voice note stored as a pxt.Audio column, lead tiering,
and per-row treg cost from billing receipts (OpenAI charges are separate).

Layout: UDFs live in udfs.py; this file holds table models, @pxt.query functions,
FastAPIRouter routes, and the interactive Lead Studio UI at GET /.

Run locally:
    export TREG_TOKEN=treg_...
    export OPENAI_API_KEY=sk-...
    pxt schema update app.py leads
    pxt service run app.py leads --port 8000
"""

# ruff: noqa: F821, RUF012

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi.responses import HTMLResponse
from udfs import (
    lead_tier,
    outreach_copywriter,
    outreach_prompt,
    treg_extract_site,
    treg_find_email,
    treg_voice_note,
    voice_note_audio,
)

import pixeltable as pxt
from pixeltable.serving import FastAPIRouter

UI_HTML = Path(__file__).resolve().parent / 'ui.html'

# ---- tables ----
TableModel = pxt.model_base()


class Leads(TableModel, name='leads', has_default_idxs=False):
    lead_id = pxt.Column(type=pxt.UUID, primary_key=True)
    full_name: pxt.String
    domain: pxt.String
    role: pxt.String | None
    stage: pxt.String  # new / queued / sent / replied

    contact = treg_find_email(lead_id, full_name, domain)
    email = contact.email.astype(pxt.String | None)
    verdict = contact.verdict.astype(pxt.String | None)
    served_by = contact.served_by.astype(pxt.String | None)

    site = treg_extract_site(lead_id, domain)
    site_markdown = site.markdown.astype(pxt.String)

    tier = lead_tier(verdict, role)

    prompt = outreach_prompt(full_name, role, domain, site_markdown)
    llm_response = outreach_copywriter(prompt, model='gpt-4o-mini')
    cold_email = llm_response.choices[0].message.content.astype(pxt.String)

    voice = treg_voice_note(lead_id, cold_email)
    voice_note = voice_note_audio(voice)

    cost_usd = (
        contact.cost_micro.astype(pxt.Float) + site.cost_micro.astype(pxt.Float) + voice.cost_micro.astype(pxt.Float)
    ) / 1_000_000.0

    __indexes__ = [pxt.BtreeIndex(domain), pxt.BtreeIndex(verdict), pxt.BtreeIndex(stage), pxt.BtreeIndex(tier)]


# ---- queries ----
@pxt.query
def all_leads(limit: int = 50) -> pxt.Query:
    """All enriched leads, ordered by name (caller-chosen UUIDs carry no recency order)."""
    return (
        Leads.order_by(Leads.full_name, Leads.lead_id)
        .limit(limit)
        .select(
            Leads.lead_id,
            Leads.full_name,
            Leads.domain,
            Leads.role,
            Leads.stage,
            Leads.email,
            Leads.verdict,
            Leads.served_by,
            Leads.tier,
            Leads.cold_email,
            Leads.voice_note,
            Leads.cost_usd,
        )
    )


@pxt.query
def by_verdict(verdict: str) -> pxt.Query:
    """Leads matching a mailbox verification verdict (verdict index)."""
    return (
        Leads.where(Leads.verdict == verdict)
        .order_by(Leads.full_name, Leads.lead_id)
        .select(
            Leads.lead_id,
            Leads.full_name,
            Leads.domain,
            Leads.role,
            Leads.stage,
            Leads.email,
            Leads.verdict,
            Leads.served_by,
            Leads.tier,
            Leads.cold_email,
            Leads.voice_note,
            Leads.cost_usd,
        )
    )


@pxt.query
def by_tier(tier: str, stage: str) -> pxt.Query:
    """Leads in a readiness tier and outreach stage (tier + stage indexes)."""
    return (
        Leads.where((Leads.tier == tier) & (Leads.stage == stage))
        .order_by(Leads.full_name)
        .select(
            Leads.lead_id,
            Leads.full_name,
            Leads.domain,
            Leads.role,
            Leads.stage,
            Leads.email,
            Leads.verdict,
            Leads.served_by,
            Leads.tier,
            Leads.cold_email,
            Leads.voice_note,
            Leads.cost_usd,
        )
    )


@pxt.query
def by_domain(domain: str) -> pxt.Query:
    """Detail for one company domain (domain index)."""
    return Leads.where(Leads.domain == domain).select(
        Leads.lead_id,
        Leads.full_name,
        Leads.domain,
        Leads.role,
        Leads.stage,
        Leads.email,
        Leads.verdict,
        Leads.served_by,
        Leads.tier,
        Leads.site_markdown,
        Leads.cold_email,
        Leads.voice_note,
        Leads.cost_usd,
    )


@pxt.query
def lead_voice_note(lead_id: uuid.UUID) -> pxt.Query:
    """Stream the generated MP3 voice note for one lead."""
    return Leads.where(Leads.lead_id == lead_id).select(Leads.voice_note)


# ---- routes ----
leads_api = FastAPIRouter(name='leads_api')


@leads_api.get('/', response_class=HTMLResponse, include_in_schema=False)
def lead_studio_ui() -> HTMLResponse:
    """Serve the interactive Lead Enrichment & Outreach Studio UI."""
    html = UI_HTML.read_text(encoding='utf-8')
    if os.environ.get('LEAD_ENRICHMENT_DEMO') == '1':
        html = html.replace(
            '<main class="shell">',
            '<main class="shell"><p class="notice visible" role="status">'
            'Demo mode: synthetic data, unverified contacts, silent audio, and no provider charges.</p>',
        )
    return HTMLResponse(html)


leads_api.add_insert_route(
    Leads,
    path='/leads',
    inputs=[Leads.lead_id, Leads.full_name, Leads.domain, Leads.role, Leads.stage],
    outputs=[
        Leads.lead_id,
        Leads.email,
        Leads.verdict,
        Leads.served_by,
        Leads.tier,
        Leads.cold_email,
        Leads.voice_note,
        Leads.cost_usd,
    ],
)
leads_api.add_compute_route(
    Leads,
    path='/preview',
    inputs=[Leads.lead_id, Leads.full_name, Leads.domain, Leads.role, Leads.stage],
    outputs=[
        Leads.email,
        Leads.verdict,
        Leads.served_by,
        Leads.tier,
        Leads.cold_email,
        Leads.voice_note,
        Leads.cost_usd,
    ],
)
leads_api.add_update_route(
    Leads, path='/leads/stage', inputs=[Leads.stage], outputs=[Leads.lead_id, Leads.full_name, Leads.stage, Leads.tier]
)
leads_api.add_delete_route(Leads, path='/leads/delete')
leads_api.add_query_route(path='/leads/all', query=all_leads, method='get')
leads_api.add_query_route(path='/leads/by-verdict', query=by_verdict, method='get')
leads_api.add_query_route(path='/leads/by-tier', query=by_tier, method='get')
leads_api.add_query_route(path='/leads/by-domain', query=by_domain, method='get')
leads_api.add_query_route(
    path='/leads/voice-note', query=lead_voice_note, method='get', one_row=True, return_fileresponse=True
)
