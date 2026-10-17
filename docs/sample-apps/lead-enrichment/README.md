<!-- pixeltable-example-app: lead-enrichment -->
# Lead Enrichment & Outreach Ledger (Clay + Lemlist in One `app.py`)

[![Built with Pixeltable](https://img.shields.io/badge/built%20with-Pixeltable-5b4bff)](https://pixeltable.com)
[![PyPI - pixeltable](https://img.shields.io/pypi/v/pixeltable?label=pixeltable)](https://pypi.org/project/pixeltable/)
[![GitHub stars](https://img.shields.io/github/stars/pixeltable/pixeltable?style=social)](https://github.com/pixeltable/pixeltable)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](../../../LICENSE)

A programmable Clay + Lemlist alternative built with Pixeltable `TableModel` computed columns, `FastAPIRouter`, an interactive web UI (`ui.html`), and custom `@pxt.udf` wrappers over [`treg.to`](https://treg.to).

Insert a `{full_name, domain}` row (or click **Enrich & Persist** in the web UI at `http://localhost:8000/`) and the `Leads` table automatically computes:
1. **Waterfall email discovery + inline mailbox verification** (`treg.people.email.find` with `X-Treg-Route-Verify: true` and a `$0.03` pre-call ceiling via `X-Treg-Route-Max-Cost`).
2. **Live company website extraction** (`treg.web.extract`) so outreach references what the company actually builds.
3. **Personalized 3-sentence cold email** (`pixeltable.functions.openai.chat_completions`).
4. **Personalized AI audio voice note** (`fishaudio.tts.s2-1-pro`), written to `TempStore` so Pixeltable persists it in `LocalStore` locally (or the database's managed home bucket on Pixeltable Cloud) and streams it at `GET /leads/voice-note?domain=...`.
5. **Deterministic lead tier & exact USD cost** (`tier`, `cost_usd`), derived from the verification verdict, scraped markdown depth, and `X-Treg-Cost-Micro` response headers with deterministic `Idempotency-Key` headers so retries never double-charge.

## What this example shows

- **Importable UDF module (`udfs.py`)**: custom `@pxt.udf` functions live in `udfs.py` and are imported by `app.py`, so the daemon, serving workers, and Pixeltable Cloud resolve them by module path (`udfs.*`)
- **Declarative schema + B-tree indexes (`app.py`)**: `Leads(TableModel, name='leads', has_default_idxs=False)` with explicit `__indexes__` on `domain`, `verdict`, `stage`, and `tier`
- **Multimodal media computed column (`pxt.Audio`)**: custom UDF writing MP3 bytes to `TempStore.create_path(extension='.mp3')` and streaming via `return_fileresponse=True`
- **All 5 `FastAPIRouter` route kinds**: `add_insert_route` (`/leads`), `add_compute_route` (`/preview` stateless dry-run), `add_update_route` (`/leads/stage`), `add_delete_route` (`/leads/delete`), and `add_query_route` (`/leads/all`, `/leads/by-verdict`, `/leads/by-tier`, `/leads/by-domain`, `/leads/voice-note`)
- **Interactive Web UI (`ui.html` at `GET /`)**: inspect enriched leads, run stateless dry-run previews without storing rows, play `pxt.Audio` voice notes inline, update outreach stages (`new` -> `contacted` -> `replied`), copy outreach emails, and inspect scraped site markdown
- **Local + Cloud config (`pixeltable.toml`)**: declares the local `leads` database and a `pxt://<your-org>:<your-db>` Cloud database with sizing and deploy excludes

## What's inside

| File | Purpose |
|------|---------|
| `.devcontainer/devcontainer.json` | GitHub Codespaces / Dev Container config (Python 3.12, installs `requirements.txt`, forwards port 8000) |
| `app.py` | `Leads` `TableModel`, `@pxt.query` functions, `FastAPIRouter` routes, and `GET /` UI handler |
| `udfs.py` | Custom `@pxt.udf` functions (`treg_find_email`, `treg_extract_site`, `treg_voice_note`, `lead_tier`) |
| `ui.html` | Interactive Lead Enrichment & Outreach Ledger served at `http://localhost:8000/` |
| `seed.py` | Idempotent seed script (`python seed.py leads`) |
| `client_demo.py` | Standard-library Python client exercising every route (`/preview`, `/leads`, queries, MP3 download) |
| `pixeltable.toml` | Local + Pixeltable Cloud database configuration |
| `requirements.txt` / `pyproject.toml` | Dependencies (`pixeltable[serve]>=0.7.16`, `httpx>=0.27`, `openai>=1.40`) |

**Tables**

| Table | Stored columns | Computed columns |
|-------|----------------|------------------|
| `leads` | `full_name`, `domain`, `stage`, `notes` | `lead_id` (PK), `contact`, `email`, `verdict`, `site`, `site_markdown`, `tier`, `prompt`, `cold_email`, `voice_note`, `cost_usd` |

**API routes** (service `leads_api`)

| Method | Path | Kind | Backed by | Notes |
|--------|------|------|-----------|-------|
| `GET` | `/` | custom | `ui.html` | Interactive Web UI |
| `POST` | `/leads` | insert | `Leads` | Enrich & persist a lead |
| `POST` | `/preview` | compute | `Leads` | Stateless dry-run enrichment (writes 0 rows) |
| `POST` | `/leads/stage` | update | `Leads` | Update `stage` and `notes` by `lead_id` |
| `POST` | `/leads/delete` | delete | `Leads` | Delete a row by `lead_id` |
| `GET` | `/leads/all` | query | `all_leads` | All leads ordered by `cost_usd` desc |
| `GET` | `/leads/by-verdict` | query | `by_verdict` | Filter by `verdict` (`valid`, `risky`, `invalid`) |
| `GET` | `/leads/by-tier` | query | `by_tier` | Filter by `tier` (`tier-1-ready`, `tier-2-review`, `tier-3-unverified`) |
| `GET` | `/leads/by-domain` | query | `by_domain` | Full lead detail + `site_markdown` (`one_row=True`) |
| `GET` | `/leads/voice-note` | query | `lead_voice_note` | Streams the MP3 `voice_note` (`one_row=True, return_fileresponse=True`) |

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
uv run python seed.py leads
uv run pxt service run app.py leads --port 8000
```

Open:
- **Interactive UI:** `http://localhost:8000/`
- **OpenAPI Docs:** `http://localhost:8000/docs`

In a second terminal, exercise all HTTP routes from Python:

```bash
uv run python client_demo.py
```

Or call the routes directly with `curl`:

```bash
# Stateless dry-run preview (computes columns without storing a row)
curl -s -X POST http://localhost:8000/preview \
  -H "Content-Type: application/json" \
  -d '{"full_name": "Guillermo Rauch", "domain": "vercel.com"}'

# Enrich and persist a lead
curl -s -X POST http://localhost:8000/leads \
  -H "Content-Type: application/json" \
  -d '{"full_name": "Pierre Brunelle", "domain": "pixeltable.com", "stage": "new"}'

# Filter by verification verdict or readiness tier
curl -s 'http://localhost:8000/leads/by-verdict?verdict=valid'
curl -s 'http://localhost:8000/leads/by-tier?tier=tier-1-ready'

# Download the personalized MP3 voice note
curl -s -o voice_note.mp3 'http://localhost:8000/leads/voice-note?domain=pixeltable.com'
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

## Deploy to Pixeltable Cloud

The same `app.py` and `udfs.py` deploy to [Pixeltable Cloud](https://pixeltable.com). Edit the `pxt://<your-org>:<your-db>` entry in `pixeltable.toml`, set your secrets, and deploy:

```bash
uv run pxt login
uv run pxt db update pxt://<your-org>:<your-db> -f
uv run pxt secret set pxt://<your-org> TREG_TOKEN=treg_... OPENAI_API_KEY=sk-...
uv run pxt db restart pxt://<your-org>:<your-db>            # pods read secrets at startup
uv run pxt schema update app.py pxt://<your-org>:<your-db> -f
uv run pxt service update app.py pxt://<your-org>:<your-db> -f
uv run pxt service list pxt://<your-org>:<your-db>
```

## Learn more

- **Awesome Pixeltable Apps gallery:** https://pierrebrunelle.github.io/awesome-pixeltable-apps/ ([GitHub repo](https://github.com/pierrebrunelle/awesome-pixeltable-apps))
- **Pixeltable Starter Kit:** https://github.com/pixeltable/pixeltable-starter-kit
- **Pixeltable Docs:** https://docs.pixeltable.com
