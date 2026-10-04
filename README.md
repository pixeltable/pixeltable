> **Pixeltable Cloud is in Limited Beta.** Email [contact@pixeltable.com](mailto:contact@pixeltable.com) if you are interested.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/release/_logo/pxt-dark.svg">
  <img alt="Pixeltable" src="https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/release/_logo/pxt-light.svg" width="40%">
</picture>

# The backend agents build with
### Multimodal database, orchestration, and serving in one file

[**Quickstart**](https://docs.pixeltable.com/overview/quick-start) |
[**Documentation**](https://docs.pixeltable.com/) |
[**CLI**](https://docs.pixeltable.com/platform/cli) |
[**Cloud**](https://docs.pixeltable.com/cloud) |
[**Discord**](https://discord.gg/QPyqFYx2UN)

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
[![tests status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml)
[![nightly status](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml)
[![stress-tests status](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml)
[![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)
[![Python](https://img.shields.io/pypi/pyversions/pixeltable)](https://pypi.org/project/pixeltable/)

[![Open in Cursor][cursor-badge]][cursor-prompt]
[![Open in Claude][claude-badge]][claude-prompt]
[![Open in ChatGPT][chatgpt-badge]][chatgpt-prompt]

Install into your agent: [Agent Skill](https://github.com/pixeltable/pixeltable-skill) ·
[MCP server](https://github.com/pixeltable/mcp-server-pixeltable-developer) ·
[Cursor plugin](https://cursor.directory/plugins/pixeltable) ·
[Claude plugin](https://claude.com/marketplace/plugins/pixeltable) ·
[ChatGPT plugin](https://chatgpt.com/plugins/plugins_6aa9bce6481c81918d73be74740eed28)

Reading this as an agent? Start at [get-started.md](https://www.pixeltable.com/get-started.md), then
[skill.md](https://docs.pixeltable.com/skill.md) and
[llms-full.txt](https://docs.pixeltable.com/llms-full.txt).

Pixeltable is the database, orchestration, and serving layers. Images, video, audio, and documents live in tables. A transform is a computed column. An index is a declaration, and so is an HTTP route. Insert a row and everything below it runs. Object storage, a vector database, an orchestrator, and the endpoint code that copies between them collapse into one application file. That file is what a coding agent can hold: it writes the whole application at once, and you review the change in one diff instead of across four systems.

```bash
pip install 'pixeltable[serve]'
pxt init
pxt service example --out app.py
pxt schema update app.py my_app
pxt service update app.py my_app
```

`pxt schema update` creates the catalog `my_app` and its tables; it does not start HTTP.
`pxt service update` starts HTTP; it does not create tables.
Generate TypeScript types from a service's OpenAPI schema and call it from a Next.js backend using the [HTTP serving guide](https://docs.pixeltable.com/howto/deployment/serving).

`pxt service example` writes this application file.

```python
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


@pxt.udf                                        # a Python function the columns below can call
def excerpt(text: str, n: int = 12) -> str:
    return text if len(text) <= n else f'{text[:n]}...'


class Docs(TableModel, name='docs'):
    id = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)  # a generated key: provided automatically on insert
    title: pxt.String
    body: pxt.String | None
    title_upper = pxtf.string.upper(title)      # an assignment: computed on insert and on update
    summary = excerpt(title)                    # a computed column over a udf this file defines


ingest = FastAPIRouter(name='ingest')
ingest.add_insert_route(                        # POST /docs inserts and returns the computed columns
    Docs, path='/docs', inputs=[Docs.title, Docs.body], outputs=[Docs.id, Docs.title_upper, Docs.summary]
)
ingest.add_update_route(                        # POST /docs/update takes id plus the new values
    Docs, path='/docs/update', inputs=[Docs.title], outputs=[Docs.id, Docs.title_upper]
)
ingest.add_compute_route(Docs, path='/titles', inputs=[Docs.title], outputs=[Docs.title_upper])
```

The same file holds `pxt.Image`, `pxt.Video`, `pxt.Audio`, or `pxt.Document` columns, and a
computed column over one of them is another assignment:
[media pipelines](https://docs.pixeltable.com/use-cases/media-processing),
[RAG](https://docs.pixeltable.com/use-cases/multimodal-backend). The port is assigned, so read it
back rather than hardcoding it:

```bash
URL=$(pxt service list --json | jq -r '.[0].endpoint')
curl -X POST "$URL/docs" \
  -H 'Content-Type: application/json' \
  -d '{"title": "Hello", "body": "world"}'
# {"id":"...","title_upper":"HELLO","summary":"Hello"}
```

The same file runs on [Pixeltable Cloud](https://docs.pixeltable.com/cloud). Sign in with `pxt login`. If `pxt org list` shows no organization, `pxt org create NAME` makes one and provisions its database, `main`. Add this entry to the project file, with your organization's name in place of `org`:

```toml
[[pixeltable.database]]
name = 'pxt://org:main'
```

```bash
pxt db update pxt://org:main
pxt schema update app.py pxt://org:main
pxt service update app.py pxt://org:main
```

The [Cloud guide](https://docs.pixeltable.com/cloud) covers the first deployment step by step, API keys, secrets, and what to run after a change.

## Chat agent or video search

[`uvx pixeltable-new`](https://github.com/pixeltable/pixeltable-new) copies one app from the [starter kit](https://github.com/pixeltable/pixeltable-starter-kit). The default copy is a chat app; pass `agent` as the last argument to `pxt schema update`. `--video` copies video search; pass `videointel`.

```bash
uvx pixeltable-new myapp
cd myapp
uv sync
pxt schema update app.py agent
pxt service update app.py agent
```

Inserting into the knowledge table needs no API key. The `/ask` route needs `ANTHROPIC_API_KEY`.

To mount the routes on an existing FastAPI app, `app.include_router(...)`. [HTTP serving](https://docs.pixeltable.com/howto/deployment/serving). To skip endpoints, run `pxt schema update`, insert from Python, then `export_sql`. [Self-hosting](https://docs.pixeltable.com/howto/deployment/overview).

## Coding agents

The links above offer the Skill, plugin, and MCP setup options. How the Skill writes `app.py`: [AI coding agents](https://docs.pixeltable.com/overview/building-pixeltable-with-llms).

The [Pixeltable plugin](https://github.com/pixeltable/pixeltable-skill) packages the Skill for Claude, Cursor, and ChatGPT. [Docs MCP](https://docs.pixeltable.com/mcp) searches public documentation; the site also offers a remote HTTP [Site MCP](https://www.pixeltable.com/mcp) for docs search and integration listing. [Hosted Cloud MCP](https://www.pixeltable.com/mcp/cloud) reads your signed-in Cloud organization through OAuth. It also starts and stops databases, restarts services, and sets secrets. The [local developer MCP](https://github.com/pixeltable/mcp-server-pixeltable-developer) runs on your machine with catalog, query, and REPL tools. Hosted Cloud MCP does not deploy or delete; use the `pxt` CLI for that.

```bash
npx skills add pixeltable/pixeltable-skill
```

The skill writes a `TableModel` in `app.py`. If the agent writes `create_table` in application code, names the file `schema.py`, or writes the removed command `pxt serve`, the installed skill is stale: reinstall `npx skills add pixeltable/pixeltable-skill`.

Notebooks and tests still use `pxt.create_table()`. An app puts tables in `app.py` and creates them with `pxt schema update`.

## License

Apache 2.0. [Contributing](https://github.com/pixeltable/pixeltable/blob/main/CONTRIBUTING.md) · [Discord](https://discord.gg/QPyqFYx2UN)

[cursor-badge]: https://img.shields.io/badge/Open_in-Cursor-000000
[claude-badge]: https://img.shields.io/badge/Open_in-Claude-D97757
[chatgpt-badge]: https://img.shields.io/badge/Open_in-ChatGPT-10A37F
[cursor-prompt]: https://cursor.com/link/prompt?text=Build+a+multimodal+AI+data+app+with+Pixeltable.+Follow+https%3A%2F%2Fpixeltable.com%2Fget-started.md.
[claude-prompt]: https://claude.ai/new?q=Build+a+multimodal+AI+data+app+with+Pixeltable.+Follow+https%3A%2F%2Fpixeltable.com%2Fget-started.md.
[chatgpt-prompt]: https://chatgpt.com/?prompt=Build+a+multimodal+AI+data+app+with+Pixeltable.+Follow+https%3A%2F%2Fpixeltable.com%2Fget-started.md.
