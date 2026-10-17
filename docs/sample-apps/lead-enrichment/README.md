# Lead Enrichment & Outreach Engine (Clay + Lemlist in One `app.py`)

A programmable lead-enrichment and multimodal outreach pipeline in a single application file ([`app.py`](app.py)) using Pixeltable `TableModel` computed columns, `FastAPIRouter`, and custom `@pxt.udf` wrappers over [`treg.to`](https://treg.to).

Insert a `{full_name, domain}` row and the table automatically runs:
1. **Waterfall email discovery + inline mailbox verification** (`treg.people.email.find` with `X-Treg-Route-Verify: true` and a `$0.03` pre-call ceiling via `X-Treg-Route-Max-Cost`).
2. **Live company website extraction** (`treg.web.extract`) to ground outreach in what the company actually builds.
3. **Personalized 3-sentence cold email** (`pixeltable.functions.openai.chat_completions`).
4. **Personalized AI audio voice note** (`fishaudio.tts.s2-1-pro`), written to `TempStore` so Pixeltable persists it in `LocalStore` locally or your database's managed home bucket on Cloud, and served as a `/api/_pxt/media/...` URL.
5. **Exact per-lead enrichment cost** (`cost_usd`), computed via native column arithmetic from `X-Treg-Cost-Micro` response headers with deterministic `Idempotency-Key` headers so retries never double-charge.

## Quickstart (Local)

Export your keys **before** the first `pxt` command — the Pixeltable daemon reads the environment once at startup, and local services inherit the daemon's environment ([get a `treg` token](https://treg.to/llms.txt)):

```bash
export TREG_TOKEN=treg_...
export OPENAI_API_KEY=sk-...

cd docs/sample-apps/lead-enrichment
uv sync
uv run pxt schema check app.py
uv run pxt service check app.py
uv run pxt schema update app.py leads
uv run pxt service update app.py leads -f
uv run pxt service list
```

### Enrich a Lead via HTTP

Resolve the service endpoint from `pxt service list` and POST a lead:

```bash
URL=$(uv run pxt service list leads --json | jq -r '.[] | select(.name == "leads") | .endpoint')

curl -s -X POST "$URL/api/leads" \
  -H "Content-Type: application/json" \
  -d '{"full_name": "Pierre Brunelle", "domain": "pixeltable.com"}'
```

The response includes `uuid`, `email`, `verdict`, `cold_email`, `voice_note` (a relative media URL under `/api/_pxt/media/...` playable at `"$URL<voice_note>"`), and `cost_usd`.

Query only verified (`verdict == "valid"`) leads:

```bash
curl -s "$URL/api/leads/verified?limit=20"
```

Delete a lead by primary key (`uuid`):

```bash
curl -s -X POST "$URL/api/delete/lead" \
  -H "Content-Type: application/json" \
  -d '{"uuid": "<uuid-from-insert>"}'
```

## Recovery & Iteration

- **Exported or changed a key after starting `pxt` (`409` / `STALE_CONFIG`):**
  ```bash
  uv run pxt daemon restart
  uv run pxt service update app.py leads -f
  ```
- **Inspect or retry failed cells without re-running succeeded rows:**
  ```bash
  uv run pxt errors leads/leads
  uv run pxt recompute leads/leads contact --errors-only -f
  ```
- **Edited a computed column expression in `app.py`:**
  ```bash
  uv run pxt schema update app.py leads
  uv run pxt recompute leads/leads cold_email -f
  uv run pxt service update app.py leads -f
  ```

## Without HTTP (Python / REPL)

```bash
uv run pxt schema update app.py leads
```

```python
import app

app.TableModel.bind_all('leads')
app.Leads.insert(
    [{'full_name': 'Pierre Brunelle', 'domain': 'pixeltable.com'}],
    on_error='ignore',
)
print(
    app.Leads.select(
        app.Leads.full_name,
        app.Leads.email,
        app.Leads.verdict,
        app.Leads.cold_email,
        app.Leads.voice_note,
        app.Leads.cost_usd,
    ).collect()
)
```

## Same `app.py` on Pixeltable Cloud

```bash
uv run pxt db update pxt://org:mydb -f
uv run pxt secret set pxt://org TREG_TOKEN=treg_... OPENAI_API_KEY=sk-...
uv run pxt db restart pxt://org:mydb        # pods read secrets at startup
uv run pxt schema update app.py pxt://org:mydb -f
uv run pxt service update app.py pxt://org:mydb -f
```

## Schema Overview

| Column | Type | Source |
|---|---|---|
| `uuid` | `pxt.UUID` (PK) | `pxtf.uuid.uuid7()` |
| `full_name`, `domain` | `pxt.String` | Stored input columns |
| `contact` | `pxt.Json` | `@pxt.udf` `treg_find_email()` (`/call/treg.people.email.find` + `X-Treg-Route-Verify: true`) |
| `email`, `verdict` | `pxt.String \| None` | Extracted from `contact.email` and `contact.verdict` |
| `site`, `site_markdown` | `pxt.Json`, `pxt.String` | `@pxt.udf` `treg_extract_site()` (`/call/treg.web.extract`) |
| `prompt` | `pxt.String` | `pxtf.string.format(...)` over `full_name`, `domain`, and `site_markdown` |
| `cold_email` | `pxt.String` | `openai.chat_completions` (`gpt-4o-mini`) |
| `voice_note` | `pxt.Audio` | `@pxt.udf` `treg_voice_note()` (`/call/fishaudio.tts.s2-1-pro` written to `TempStore`) |
| `cost_usd` | `pxt.Float` | `(contact.cost_micro + site.cost_micro) / 1_000_000.0` |
