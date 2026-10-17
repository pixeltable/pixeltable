<!-- pixeltable-example-app: lead-enrichment -->
# Lead enrichment and outreach ledger

Insert a prospect to find and verify a work email with [treg.to](https://treg.to), extract company website text, draft an email with OpenAI, and generate a Fish Audio voice note. Pixeltable stores the results as computed columns and serves them through `FastAPIRouter` and the included web UI. This sample drafts outreach; it does not send it.

Each insert and preview spends your treg.to prepaid balance and calls OpenAI, which bills separately. The seed script enriches three prospects. `cost_usd` sums the email, website, and voice-note receipts for the current results. It excludes OpenAI and previous recomputations, so it is not a cumulative spend ledger.

## What this example shows

- `app.py` declares the `Leads` table, four B-tree indexes, queries, and insert, compute, update, delete, and query routes.
- `udfs.py` keeps custom UDFs importable by local and hosted workers.
- `voice` stores an audio receipt, including base64 MP3 bytes and the treg charge. `voice_note_audio()` writes those bytes through `TempStore` into a `pxt.Audio` column. Recomputing that column alone does not synthesize again.
- The UI at `GET /` supports enrichment, paid previews, stage updates (`new`, `queued`, `sent`, `replied`), audio playback, copying email drafts, and reading website context.
- `pixeltable.toml` has local and hosted database entries. Replace the hosted placeholder before deployment.

## Tables and routes

| Table | Stored inputs | Computed columns |
|-------|---------------|------------------|
| `leads` | `lead_id` (UUID primary key), `full_name`, `domain`, `role`, `stage` | `contact`, `email`, `verdict`, `served_by`, `site`, `site_markdown`, `tier`, `prompt`, `llm_response`, `cold_email`, `voice`, `voice_note`, `cost_usd` |

Supply a new `lead_id` for new work. Keep that ID and the same input on retries of an aborted HTTP insert. treg keys include this ID, the endpoint, and the entire request. treg retains keys for 24 hours; after that, repeating a request can charge again. Changing input or previewing with a new ID is new work. A replay's original charge is retained in the receipt, rather than replacing it with zero. See [treg's retry contract](https://treg.to/llms.txt).

| Method | Path | Result |
|--------|------|--------|
| `GET` | `/` | Web UI |
| `POST` | `/leads` | Enrich and store one lead |
| `POST` | `/preview` | Paid enrichment, including audio, without saving a lead |
| `POST` | `/leads/stage` | Update `stage` by `lead_id` |
| `POST` | `/leads/delete` | Delete by `lead_id` |
| `GET` | `/leads/all?limit=50` | List ordered by descending `lead_id` |
| `GET` | `/leads/by-verdict?verdict=valid` | List filtered by verification verdict |
| `GET` | `/leads/by-tier?tier=priority&stage=new` | List filtered by tier (`priority`, `ready`, `review`, `skip`) and required stage |
| `GET` | `/leads/by-domain?domain=pixeltable.com` | List of all matching leads, including website context |
| `GET` | `/leads/voice-note?lead_id=...` | MP3 for that lead |

Query routes return lists in a `rows` envelope. Insert, preview, and update return one object. JSON media fields contain absolute URLs. Use `voice_note` directly for playback; hosted media URLs expire, so query again to refresh them.

## Run locally

Export keys before the first `pxt` command. The daemon reads its environment at startup. [Create a treg token](https://treg.to).

```bash
export TREG_TOKEN=your-treg-token
export OPENAI_API_KEY=your-openai-key

cd docs/sample-apps/lead-enrichment
uv sync
uv run pxt schema check app.py
uv run pxt service check app.py
uv run pxt schema update app.py leads
uv run pxt service run app.py leads --port 8000
```

Open [the UI](http://localhost:8000/) or [OpenAPI docs](http://localhost:8000/docs). In a second terminal, optionally run `uv run python seed.py leads` or `uv run python client_demo.py`. The client makes separate paid preview and insert calls, queries the results, downloads audio, updates the stage, and deletes its inserted row.

For an HTTP insert, choose a UUID once and keep it for retries:

```bash
curl --fail-with-body -X POST http://localhost:8000/leads \
  -H 'Content-Type: application/json' \
  -d '{"lead_id":"8156fb2c-801a-4b8a-858e-3fed50860b0a","full_name":"Pierre Brunelle","domain":"pixeltable.com","role":"Founder","stage":"new"}'

curl --fail-with-body 'http://localhost:8000/leads/by-tier?tier=priority&stage=new'
curl --fail-with-body -o voice_note.mp3 \
  'http://localhost:8000/leads/voice-note?lead_id=8156fb2c-801a-4b8a-858e-3fed50860b0a'
```

## Recover and iterate

After changing exported keys, restart the daemon and service:

```bash
uv run pxt daemon restart
uv run pxt service update app.py leads
```

A failed HTTP insert rolls back its row. Inspect the error and retry the same POST with the same UUID and unchanged inputs. A pending response is polled through its catalog descriptor, rather than submitted again. An unsupported provider-native terminal shape fails explicitly instead of storing an empty result. Pending BounceBan verification is polled separately and its settled charge is included. If polling times out or a verifier cannot be resumed, inspect the call ID in the error through treg before starting new work: the task may still complete and charge.

The seed script uses `on_error='ignore'`, which retains rows with failed cells. For those rows, inspect errors and recompute the failed column, for example:

```bash
uv run pxt errors leads/leads
uv run pxt recompute leads/leads contact --errors-only
```

After editing a computed expression, update the schema and recompute the column you edited. Dependents refresh automatically. For example, editing `prompt` requires:

```bash
uv run pxt schema update app.py leads
uv run pxt recompute leads/leads prompt
uv run pxt service update app.py leads
```

Editing the model in `llm_response` instead requires recomputing `llm_response`. These operations call OpenAI again for every selected row and cascade to `voice` and `voice_note`, which can incur another treg charge for changed audio input. Recomputing `cold_email` only extracts the stored response; it does not call OpenAI again. The CLI asks for confirmation; `-f` skips that cost confirmation.

## Deploy to Pixeltable Cloud

Replace `pxt://<your-org>:<your-db>` in `pixeltable.toml` with the URI of an existing hosted database, for example `pxt://acme:main`. `pxt db update` requires that matching named entry. Use the same real URI in these commands:

```bash
uv run pxt login
uv run pxt secret set pxt://acme TREG_TOKEN=your-treg-token OPENAI_API_KEY=your-openai-key
uv run pxt db update pxt://acme:main
uv run pxt db restart pxt://acme:main
uv run pxt schema update app.py pxt://acme:main
uv run pxt service update app.py pxt://acme:main
uv run pxt service list pxt://acme:main
```

Pods read secrets at startup. For database creation and authenticated service access, see [Pixeltable Cloud](https://docs.pixeltable.com/cloud) and [HTTP serving](https://docs.pixeltable.com/howto/deployment/serving). The bundled UI is for the unauthenticated local service; use the API-key-aware Python client for hosted endpoints.

## Learn more

- [Awesome Pixeltable Apps](https://github.com/pierrebrunelle/awesome-pixeltable-apps) and its [gallery](https://pierrebrunelle.github.io/awesome-pixeltable-apps/)
- [Pixeltable Starter Kit](https://github.com/pixeltable/pixeltable-starter-kit)
- [Pixeltable documentation](https://docs.pixeltable.com)
