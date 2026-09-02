<picture class="github-only">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/e9bf82b2-cace-4bd8-9523-b65495eb8131">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/c5ab123e-806c-49bf-93e7-151353719b16">
  <img alt="Pixeltable Logo" src="https://github.com/user-attachments/assets/e9bf82b2-cace-4bd8-9523-b65495eb8131" width="40%">
</picture>

<div>
<br>
</div>

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
[![tests status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml)
[![nightly status](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml)
[![stress-tests status](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml)
[![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)
[![Python](https://img.shields.io/pypi/pyversions/pixeltable)](https://pypi.org/project/pixeltable/)

[**Quickstart**](https://docs.pixeltable.com/overview/quick-start) |
[**Documentation**](https://docs.pixeltable.com/) |
[**Cloud**](https://docs.pixeltable.com/howto/deployment/cloud) |
[**Skill**](https://github.com/pixeltable/pixeltable-skill) |
[**llms-full.txt**](https://docs.pixeltable.com/llms-full.txt) |
[**Discord**](https://discord.gg/QPyqFYx2UN)

## One application file

Most stacks glue a blob store, a warehouse, a vector database, an orchestrator, and custom endpoints. You pay for the joints. Pixeltable is the database, the orchestration, and the serving: tables, computed columns, indexes, and endpoints in one Python file (`app.py`). Insert a row. Transforms run.

Locally, `pxt schema update` creates the tables. `pxt service update` starts the endpoints. On Cloud, use the same file after you set `PIXELTABLE_API_KEY` (create the key in the [dashboard](https://docs.pixeltable.com/howto/deployment/cloud#get-an-api-key)) and run `pxt db update`. Python 3.11+ on Linux, macOS, or Windows.

```bash
pip install 'pixeltable[serve]'
pxt init
pxt service example --out app.py
pxt schema update app.py my_app
pxt service update app.py my_app
```

`pxt service example` writes this file:

```python
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


class Docs(TableModel, name='docs'):
    title: pxt.String  # a stored column
    body: pxt.String | None  # a stored column that may be null
    title_upper = pxtf.string.upper(title)  # computed: an assignment, not an annotation


ingest = FastAPIRouter(name='ingest')
ingest.add_insert_route(
    Docs,
    path='/docs',
    inputs=[Docs.title, Docs.body],
    outputs=[Docs.title, Docs.title_upper],
)
```

`title: pxt.String` is a value you insert. `title_upper = ...` is computed on insert and on update. `pxt service list` prints the URL (the port is assigned):

```bash
pxt service list
# ingest  http://127.0.0.1:<port>  ...
curl -X POST http://127.0.0.1:<port>/docs \
  -H 'Content-Type: application/json' \
  -d '{"title": "Hello", "body": "world"}'
```

`pxt schema update` creates the catalog namespace `my_app` and its tables. That is not a folder on disk. It does not start the endpoints. After the tables exist, `pxt service update` starts them.

To put the same file on Pixeltable Cloud, create an API key in the [Cloud dashboard](https://docs.pixeltable.com/howto/deployment/cloud#get-an-api-key), set `PIXELTABLE_API_KEY`, name the database in `pixeltable.toml`, then run the three commands below. `pxt db update` creates or updates the hosted database. It does not insert rows and does not start app endpoints. `pxt schema update` creates tables there. `pxt service update` starts the endpoints on the host. `pxt service run` always serves from this process and cannot target Cloud:

```bash
pxt db update pxt://org:mydb
pxt schema update app.py pxt://org:mydb
pxt service update app.py pxt://org:mydb
```

Same steps: [Quickstart](https://docs.pixeltable.com/overview/quick-start).

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

## Agents write the same file

```bash
npx skills add pixeltable/pixeltable-skill
```

Skill **2.7.0+** writes a `TableModel` in `app.py`, then `pxt schema update`. If the agent writes `create_table` in app code, `schema.py`, or `pxt serve`, the installed skill is stale: this page wins. Reinstall: `npx skills add pixeltable/pixeltable-skill`. Do not copy this repo's `AGENTS.md` into an application; that file is for Pixeltable contributors.

Notebooks and tests still use `pxt.create_table()`. An app puts tables in `app.py` and creates them with `pxt schema update`.

## License

Apache 2.0. [Contributing](CONTRIBUTING.md) · [Discord](https://discord.gg/QPyqFYx2UN)
