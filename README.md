> **Pixeltable Cloud is in Limited Beta.** Email [contact@pixeltable.com](mailto:contact@pixeltable.com) if you are interested.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/release/_logo/pxt-dark.svg">
  <img alt="Pixeltable" src="https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/release/_logo/pxt-light.svg" width="40%">
</picture>

## The unified multimodal backend for AI data apps in one Python file

[**Quickstart**](https://docs.pixeltable.com/overview/quick-start) |
[**Documentation**](https://docs.pixeltable.com/) |
[**CLI**](https://docs.pixeltable.com/platform/cli) |
[**Cloud**](https://docs.pixeltable.com/howto/deployment/cloud) |
[**Skill**](https://github.com/pixeltable/pixeltable-skill) |
[**get-started.md**](https://www.pixeltable.com/get-started.md) |
[**skill.md**](https://docs.pixeltable.com/skill.md) |
[**llms-full.txt**](https://docs.pixeltable.com/llms-full.txt) |
[**Discord**](https://discord.gg/QPyqFYx2UN)

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
[![tests status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml)
[![nightly status](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml)
[![stress-tests status](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml)
[![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)
[![Python](https://img.shields.io/pypi/pyversions/pixeltable)](https://pypi.org/project/pixeltable/)

Pixeltable is the database, orchestration, and serving layers. Images, video, audio, and documents live in tables. A transform is a computed column. An index is a declaration, and so is an HTTP route. Insert a row and everything below it runs. Object storage, a vector database, an orchestrator, and the endpoint code that copies between them collapse into one application file.

```bash
pip install 'pixeltable[serve]'
pxt init
pxt service example --out app.py
pxt schema update app.py my_app
pxt service update app.py my_app
```

`pxt schema update` creates the catalog `my_app` and its tables; it does not start HTTP.
`pxt service update` starts HTTP; it does not create tables.

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
    doc_id = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)  # a generated key: provided automatically on insert
    title: pxt.String
    body: pxt.String | None
    title_upper = pxtf.string.upper(title)      # an assignment: computed on insert and on update
    summary = excerpt(title)                    # a computed column over a udf this file defines


ingest = FastAPIRouter(name='ingest')
ingest.add_insert_route(                        # POST /docs inserts and returns the computed columns
    Docs, path='/docs', inputs=[Docs.title, Docs.body], outputs=[Docs.doc_id, Docs.title_upper, Docs.summary]
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
# {"doc_id":"...","title_upper":"HELLO","summary":"Hello"}
```

The same file runs on Pixeltable Cloud. Create an API key in the [Cloud dashboard](https://docs.pixeltable.com/howto/deployment/cloud#get-an-api-key), set `PIXELTABLE_API_KEY`, name the database in `pixeltable.toml`, then target it by URI. `pxt db update` creates or updates the hosted database; it does not insert rows. `pxt service run` is local only and cannot target Cloud.

```bash
pxt db update pxt://org:mydb
pxt schema update app.py pxt://org:mydb
pxt service update app.py pxt://org:mydb
```

A `@pxt.udf` in that same `app.py` is in the image `pxt db update` builds.

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

Hand the agent [get-started.md](https://www.pixeltable.com/get-started.md). That playbook installs the package, the skill, and MCP. How the skill writes `app.py`: [AI coding agents](https://docs.pixeltable.com/overview/building-pixeltable-with-llms). Docs capability file: [skill.md](https://docs.pixeltable.com/skill.md).

```bash
npx skills add pixeltable/pixeltable-skill
```

The skill writes a `TableModel` in `app.py`. If the agent writes `create_table` in application code, names the file `schema.py`, or writes the removed command `pxt serve`, the installed skill is stale: reinstall `npx skills add pixeltable/pixeltable-skill`.

Notebooks and tests still use `pxt.create_table()`. An app puts tables in `app.py` and creates them with `pxt schema update`.

## License

Apache 2.0. [Contributing](https://github.com/pixeltable/pixeltable/blob/main/CONTRIBUTING.md) · [Discord](https://discord.gg/QPyqFYx2UN)
