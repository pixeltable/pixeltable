"""Lead Enrichment & Multimodal Outreach API built with Pixeltable and treg.to.

Insert a prospect (full_name, domain, role, stage). Pixeltable runs a declarative
enrichment DAG: waterfall email discovery + BounceBan mailbox verification via treg.to,
live website extraction via treg.to, a personalized 3-sentence cold email via OpenAI,
a personalized Fish Audio MP3 voice note stored as a pxt.Audio column, lead tiering,
and exact per-row USD cost calculation from X-Treg-Cost-Micro headers.

Layout: UDFs live in udfs.py; this file holds table models, @pxt.query functions,
FastAPIRouter routes, and the interactive Lead Studio UI at GET /.

Run locally:
    export TREG_TOKEN=treg_...
    export OPENAI_API_KEY=sk-...
    pxt schema update app.py leads
    pxt service run app.py leads --port 8000
"""

# ruff: noqa: F821

from __future__ import annotations

from pathlib import Path

from fastapi.responses import HTMLResponse

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter
from udfs import (
    lead_tier,
    outreach_copywriter,
    treg_extract_site,
    treg_find_email,
    treg_voice_note,
)

UI_HTML = Path(__file__).resolve().parent / "ui.html"

# ---- tables ----
TableModel = pxt.model_base()


class Leads(TableModel, name="leads", has_default_idxs=False):
    lead_id = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)
    full_name: pxt.String
    domain: pxt.String
    role: pxt.String | None
    stage: pxt.String  # new / queued / sent / replied

    # 1. Waterfall email enrichment + BounceBan verification via treg.to
    contact = treg_find_email(full_name, domain)
    email = contact.email.astype(pxt.String | None)
    verdict = contact.verdict.astype(pxt.String | None)
    served_by = contact.served_by.astype(pxt.String | None)

    # 2. Live website scrape via treg.to routed web extractor
    site = treg_extract_site(domain)
    site_markdown = site.markdown.astype(pxt.String | None)

    # 3. Lead routing tier (priority / ready / review / skip)
    tier = lead_tier(verdict, role)

    # 4. Personalized cold email grounded in scraped company context
    cold_email = outreach_copywriter(full_name, role, domain, site_markdown)

    # 5. Personalized AI audio voice note stored in Pixeltable's media store
    voice_note = treg_voice_note(cold_email)

    # 6. Settled enrichment cost in USD from X-Treg-Cost-Micro
    cost_usd = (
        (
            contact.cost_micro.astype(pxt.Float | None)
            + site.cost_micro.astype(pxt.Float | None)
        )
        / 1_000_000.0
    ).astype(pxt.Float | None)

    __indexes__ = [
        pxt.BtreeIndex(domain),
        pxt.BtreeIndex(verdict),
        pxt.BtreeIndex(stage),
        pxt.BtreeIndex(tier),
    ]


# ---- queries ----
@pxt.query
def all_leads(limit: int = 50):
    """All enriched leads, most recent first."""
    return (
        Leads.order_by(Leads.lead_id, asc=False)
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
def by_verdict(verdict: str):
    """Leads matching a mailbox verification verdict (verdict index)."""
    return (
        Leads.where(Leads.verdict == verdict)
        .order_by(Leads.lead_id, asc=False)
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
def by_tier(tier: str, stage: str):
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
            Leads.cost_usd,
        )
    )


@pxt.query
def by_domain(domain: str):
    """Detail for one company domain (domain index)."""
    return (
        Leads.where(Leads.domain == domain)
        .order_by(Leads.lead_id, asc=False)
        .limit(1)
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
            Leads.site_markdown,
            Leads.cold_email,
            Leads.voice_note,
            Leads.cost_usd,
        )
    )


@pxt.query
def lead_voice_note(domain: str):
    """Stream the generated MP3 voice note for a domain."""
    return (
        Leads.where(Leads.domain == domain)
        .order_by(Leads.lead_id, asc=False)
        .limit(1)
        .select(Leads.voice_note)
    )


# ---- routes ----
leads_api = FastAPIRouter(name="leads_api")


@leads_api.get("/", response_class=HTMLResponse, include_in_schema=False)
def lead_studio_ui() -> HTMLResponse:
    """Serve the interactive Lead Enrichment & Outreach Studio UI."""
    return HTMLResponse(UI_HTML.read_text(encoding="utf-8"))


leads_api.add_insert_route(
    Leads,
    path="/leads",
    inputs=[Leads.full_name, Leads.domain, Leads.role, Leads.stage],
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
    path="/preview",
    inputs=[Leads.full_name, Leads.domain, Leads.role, Leads.stage],
    outputs=[
        Leads.email,
        Leads.verdict,
        Leads.served_by,
        Leads.tier,
        Leads.cold_email,
        Leads.cost_usd,
    ],
)
leads_api.add_update_route(
    Leads,
    path="/leads/stage",
    inputs=[Leads.stage],
    outputs=[Leads.lead_id, Leads.full_name, Leads.stage, Leads.tier],
)
leads_api.add_delete_route(Leads, path="/leads/delete")
leads_api.add_query_route(path="/leads/all", query=all_leads, method="get")
leads_api.add_query_route(path="/leads/by-verdict", query=by_verdict, method="get")
leads_api.add_query_route(path="/leads/by-tier", query=by_tier, method="get")
leads_api.add_query_route(
    path="/leads/by-domain", query=by_domain, method="get", one_row=True
)
leads_api.add_query_route(
    path="/leads/voice-note",
    query=lead_voice_note,
    method="get",
    one_row=True,
    return_fileresponse=True,
)
