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
[**CLI**](https://docs.pixeltable.com/platform/cli) |
[**Dashboard**](https://docs.pixeltable.com/platform/dashboard) |
[**Cloud**](https://docs.pixeltable.com/howto/deployment/cloud) |
[**Starter kit**](https://docs.pixeltable.com/resources/starter-kit) |
[**Skill**](https://github.com/pixeltable/pixeltable-skill) |
[**llms-full.txt**](https://docs.pixeltable.com/llms-full.txt) |
[**Discord**](https://discord.gg/QPyqFYx2UN)

## One application file

Tables, computed columns, and HTTP routes live in `app.py`. Insert runs compute. Apply locally, then the same file against `pxt://`. Python 3.11+ on Linux, macOS, or Windows.

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
    title: pxt.String                         # a stored column
    body: pxt.String | None                   # a stored column that may be null
    title_upper = pxtf.string.upper(title)    # a computed column: an assignment, not an annotation


ingest = FastAPIRouter(name='ingest')
ingest.add_insert_route(
    Docs, path='/docs', inputs=[Docs.title, Docs.body], outputs=[Docs.title, Docs.title_upper]
)
```

An annotation is a stored column. An assignment is a computed column. Insert a row; `title_upper` is computed on the way in. `pxt service list` prints the URL (the port is assigned):

```bash
pxt service list
# ingest  http://127.0.0.1:<port>  ...
curl -X POST http://127.0.0.1:<port>/docs \
  -H 'Content-Type: application/json' \
  -d '{"title": "Hello", "body": "world"}'
```

`pxt schema update` creates the directory and tables. It does not start HTTP. `pxt service update` does not create tables. Apply first.

Hosted catalog (`PIXELTABLE_API_KEY` required). Pack, declare, then serve. `pxt service run` is local only:

```bash
pxt db update pxt://org:mydb
pxt schema update app.py pxt://org:mydb
pxt service update app.py pxt://org:mydb
```

Same steps: [Quickstart](https://docs.pixeltable.com/overview/quick-start).

## Chat agent or video search

[`uvx pixeltable-new`](https://github.com/pixeltable/pixeltable-new) copies one app from the [starter kit](https://github.com/pixeltable/pixeltable-starter-kit). Default is the chat agent (catalog TARGET `agent`). `--video` is video search (`videointel`).

```bash
uvx pixeltable-new myapp
cd myapp
uv sync
pxt schema update app.py agent
pxt service update app.py agent
```

Knowledge insert needs no API key. `/ask` needs `ANTHROPIC_API_KEY`.

Already have FastAPI: `app.include_router(api)`. [HTTP serving](https://docs.pixeltable.com/howto/deployment/serving). No HTTP: insert, `export_sql`, exit. [Self-hosting](https://docs.pixeltable.com/howto/deployment/overview).

## Agents write the same file

```bash
npx skills add pixeltable/pixeltable-skill
```

The skill writes a `TableModel` in `app.py`, then `pxt schema update`. Do not copy this repo's `AGENTS.md` into an application; that file is for Pixeltable contributors.

Notebooks and tests still use `pxt.create_table()`. An application's contract is the application file.

## License

Apache 2.0. [Contributing](CONTRIBUTING.md) · [Discord](https://discord.gg/QPyqFYx2UN)
