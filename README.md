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

## Make Building Multimodal AI Data Apps Dead Simple

**Pixeltable is the unified multimodal backend for AI data apps in one Python file.** Store media, run models, index embeddings, and serve HTTP from `app.py`. Insert a row. Computed columns run. Indexes stay current. You do not glue together a blob store, a vector database, an orchestrator, and handlers.

| Declare | Insert | Serve |
| --- | --- | --- |
| `pxt schema update` | Insert a row | `pxt service update` |
| Creates the tables `app.py` declares. Media and structured columns live together. | Assigned columns run. Indexes stay current. | Starts the `FastAPIRouter` services `app.py` declares. `pxt service list` prints the URL. |

Declare first. `pxt schema update` does not start HTTP. `pxt service update` does not create tables. Hosted catalog: [Cloud](https://docs.pixeltable.com/howto/deployment/cloud) (`pxt db update` ships the project image; `pxt service` stays local).

## Insert a row, the rest follows

Expand a row for what Pixeltable replaces, a snippet that belongs in `app.py`, and docs. Snippets assume `TableModel = pxt.model_base()` and `import pixeltable.functions as pxtf`. Notebooks still use `pxt.create_table()`.

<details>
<summary><b>Store:</b> one table for media and rows</summary>
<br>

`pxt.Image`, `pxt.Video`, `pxt.Audio`, `pxt.Document`, `pxt.Json`. Not S3 + Postgres + boto3 sync.

```python
class Media(TableModel, name='media'):
    img: pxt.Image
    video: pxt.Video
    audio: pxt.Audio
    document: pxt.Document
    metadata: pxt.Json
```

[Type system](https://docs.pixeltable.com/platform/type-system) · [Tables](https://docs.pixeltable.com/tutorials/tables-and-data-operations) · [Cloud storage](https://docs.pixeltable.com/integrations/cloud-storage)
</details>

<details>
<summary><b>Iterate:</b> documents become chunks, video becomes frames</summary>
<br>

An iterator on the `TableModel` explodes one row into many. Insert a document or a video. Child rows appear. Not FFmpeg/spaCy jobs with foreign keys.

```python
class Docs(TableModel, name='docs'):
    pdf: pxt.Document


class Chunks(
    TableModel,
    name='chunks',
    base=Docs,
    iterator=pxtf.document.document_splitter(
        Docs.pdf, separators='sentence,token_limit', limit=300
    ),
):
    pass


class Videos(TableModel, name='videos'):
    video: pxt.Video


class Frames(
    TableModel,
    name='frames',
    base=Videos,
    iterator=pxtf.video.frame_iterator(Videos.video, fps=1),
):
    thumb = frame.resize((256, 256))  # type: ignore[name-defined]
```

[Views](https://docs.pixeltable.com/platform/views) · [Iterators](https://docs.pixeltable.com/platform/iterators) · [RAG cookbook](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-rag-pipeline) · [Video frames](https://docs.pixeltable.com/howto/cookbooks/video/video-extract-frames)
</details>

<details>
<summary><b>Orchestrate:</b> models are assignments</summary>
<br>

An assignment is a computed column. It runs on insert for new or stale rows only. Built-ins cover media and [30+ providers](https://docs.pixeltable.com/integrations/frameworks). Not Airflow or a full re-run.

```python
class Docs(TableModel, name='docs'):
    body: pxt.String
    summary = pxtf.openai.chat_completions(
        messages=[{'role': 'user', 'content': body}],
        model='gpt-4o-mini',
    )
```

[Computed columns](https://docs.pixeltable.com/tutorials/computed-columns) · [How it works](https://docs.pixeltable.com/overview/how-it-works)
</details>

<details>
<summary><b>Index:</b> vector search on the column</summary>
<br>

`EmbeddingIndex` stays current on insert. `.similarity()` and `.where()` are one query. Not Pinecone or Milvus plus a filter job.

```python
class Items(TableModel, name='items'):
    body: pxt.String
    __indexes__ = [
        pxt.EmbeddingIndex(
            body,
            embedding=pxtf.huggingface.sentence_transformer.using(
                model_id='sentence-transformers/all-MiniLM-L6-v2'
            ),
            name='body_idx',
        )
    ]
```

```python
items = pxt.get_table('my_app.items')
sim = items.body.similarity(string='application file')
items.order_by(sim, asc=False).limit(5).select(items.body)
```

[Embedding indexes](https://docs.pixeltable.com/platform/embedding-indexes) · [Migrate](https://docs.pixeltable.com/howto/coming-from)
</details>

<details>
<summary><b>Agents:</b> tool calls are columns</summary>
<br>

`pxt.tools()` and `invoke_tools()`. Insert a message. The choice and the result are stored. Not a LangGraph loop in memory.

```python
@pxt.udf
def get_weather(city: str) -> str:
    return f'Weather in {city}: 72°F, sunny'


tools = pxt.tools(get_weather)


class Assistant(TableModel, name='assistant'):
    message: pxt.String
    response = pxtf.openai.chat_completions(
        messages=[{'role': 'user', 'content': message}],
        model='gpt-4o-mini',
        tools=tools,
    )
    tool_output = pxtf.openai.invoke_tools(tools, response)
```

[Tool calling](https://docs.pixeltable.com/howto/cookbooks/agents/llm-tool-calling)
</details>

<details>
<summary><b>Serve:</b> HTTP from the same file</summary>
<br>

`FastAPIRouter` in `app.py`. `add_insert_route` is the insert URL. Keep `@router.post` for multi-table work. Not a hand-written handler per table.

```python
from pixeltable.serving import FastAPIRouter

ingest = FastAPIRouter(name='ingest')
ingest.add_insert_route(
    Docs, path='/docs', inputs=[Docs.title, Docs.body], outputs=[Docs.title, Docs.title_upper]
)
```

```bash
pxt schema update app.py my_app
pxt service update app.py my_app
```

Already have FastAPI: `app.include_router(api)`. No HTTP: insert, `export_sql`, exit.

[HTTP serving](https://docs.pixeltable.com/howto/deployment/serving) · [Self-hosting](https://docs.pixeltable.com/howto/deployment/overview)
</details>

<details>
<summary><b>Extend:</b> your functions, cached</summary>
<br>

`@pxt.udf` and `@pxt.query`. Call them as assignments in `app.py`.

```python
@pxt.udf
def excerpt(text: str, n: int = 12) -> str:
    return text if len(text) <= n else f'{text[:n]}...'


class Docs(TableModel, name='docs'):
    title: pxt.String
    short = excerpt(title)
```

[UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) · [Custom aggregates](https://docs.pixeltable.com/howto/cookbooks/core/custom-aggregates-uda)
</details>

<details>
<summary><b>Query:</b> select after apply</summary>
<br>

`pxt.get_table()` after `pxt schema update`. Explore with `.select()`. Ship the same expression as an assignment in `app.py`.

```python
docs = pxt.get_table('my_app.docs')
docs.where(docs.title == 'Hello').select(docs.title, docs.title_upper).collect()
```

[Queries](https://docs.pixeltable.com/tutorials/queries-and-expressions)
</details>

<details>
<summary><b>Import / export:</b> I/O without a second ETL</summary>
<br>

In a notebook: `create_table(source=...)`. After apply: `insert()` from a path and `export_parquet` / `export_sql`.

```python
pxt.create_table('scratch.data', source='data.csv')
docs = pxt.get_table('my_app.docs')
docs.insert('s3://bucket/more.parquet')
pxt.io.export_parquet(docs, 'docs.parquet')
```

[CSV import](https://docs.pixeltable.com/howto/cookbooks/data/data-import-csv) · [Hugging Face](https://docs.pixeltable.com/howto/cookbooks/data/data-import-huggingface) · [SQL export](https://docs.pixeltable.com/howto/cookbooks/data/data-export-sql)
</details>

<details>
<summary><b>Inspect:</b> errors and the local UI</summary>
<br>

Failed cells stay in the table (`errortype`, `errormsg`). `pxt dashboard` browses tables, media, and lineage.

```bash
pxt errors my_app/docs
pxt dashboard
```

[CLI](https://docs.pixeltable.com/platform/cli) · [Dashboard](https://docs.pixeltable.com/platform/dashboard) · [Observability](https://docs.pixeltable.com/platform/observability)
</details>

<details>
<summary><b>Version:</b> history and revert</summary>
<br>

Every insert and schema change is a version. `pxt schema diff app.py my_app` is the reviewed plan. Destructive ops need `--allow-destructive`.

```python
t = pxt.get_table('my_app.docs')
t.history()
t.revert()
```

[Version control](https://docs.pixeltable.com/platform/version-control) · [How it works](https://docs.pixeltable.com/overview/how-it-works)
</details>

Pinecone, Postgres, and LangGraph map onto those columns. [Migrate](https://docs.pixeltable.com/howto/coming-from).

## First run

Python 3.11+ on Linux, macOS, or Windows. venv and uv: [Quickstart](https://docs.pixeltable.com/overview/quick-start).

```bash
pip install 'pixeltable[serve]'
pxt init
pxt service example --out app.py
pxt schema update app.py my_app
pxt service update app.py my_app
```

`pxt init` writes `pixeltable.toml` (this directory is the project root). `pxt service example` writes this file (`from __future__ import annotations` is required on Python 3.14+):

```python
from __future__ import annotations  # required to declare a model on Python 3.14+

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

Insert a row. `title_upper` is computed on the way in. `pxt service list` prints the URL (the port is assigned):

```bash
pxt service list
# ingest  http://127.0.0.1:<port>  ...
curl -X POST http://127.0.0.1:<port>/docs \
  -H 'Content-Type: application/json' \
  -d '{"title": "Hello", "body": "world"}'
```

## Same file on a hosted catalog

`PIXELTABLE_API_KEY` required. `pxt service` stays local:

```bash
pxt db create pxt://org:mydb
pxt schema update app.py pxt://org:mydb
```

[Cloud](https://docs.pixeltable.com/howto/deployment/cloud).

## A larger first app

[`uvx pixeltable-new`](https://github.com/pixeltable/pixeltable-new) copies one app from the [starter kit](https://github.com/pixeltable/pixeltable-starter-kit). Default is the chat agent (catalog TARGET `agent`). `--video` is video search (`videointel`).

```bash
uvx pixeltable-new myapp
cd myapp
uv sync
pxt schema update app.py agent
pxt service update app.py agent
```

Knowledge insert needs no API key. `/ask` needs `ANTHROPIC_API_KEY`.

## Agents write the same file

```bash
npx skills add pixeltable/pixeltable-skill
```

The skill writes a `TableModel` in `app.py`, then `pxt schema update`. Do not copy this repo's `AGENTS.md` into an application; that file is for Pixeltable contributors.

## Demo

https://github.com/user-attachments/assets/b50fd6df-5169-4881-9dbe-1b6e5d06cede

## Documentation

[Why Pixeltable?](https://docs.pixeltable.com/overview/pixeltable) · [Quickstart](https://docs.pixeltable.com/overview/quick-start) · [How it works](https://docs.pixeltable.com/overview/how-it-works) · [Self-hosting](https://docs.pixeltable.com/howto/deployment/overview)

| Topic | Guides |
| --- | --- |
| Schema | [Type system](https://docs.pixeltable.com/platform/type-system) · [Computed columns](https://docs.pixeltable.com/tutorials/computed-columns) · [Views](https://docs.pixeltable.com/platform/views) · [Iterators](https://docs.pixeltable.com/platform/iterators) · [Embedding indexes](https://docs.pixeltable.com/platform/embedding-indexes) · [Version control](https://docs.pixeltable.com/platform/version-control) |
| Serve | [HTTP serving](https://docs.pixeltable.com/howto/deployment/serving) · [CLI](https://docs.pixeltable.com/platform/cli) · [Dashboard](https://docs.pixeltable.com/platform/dashboard) · [UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) · [Tool calling](https://docs.pixeltable.com/howto/cookbooks/agents/llm-tool-calling) · [RAG](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-rag-pipeline) |
| Cloud and I/O | [Cloud](https://docs.pixeltable.com/howto/deployment/cloud) · [Cloud storage](https://docs.pixeltable.com/integrations/cloud-storage) · [Configuration](https://docs.pixeltable.com/platform/configuration) · [CSV](https://docs.pixeltable.com/howto/cookbooks/data/data-import-csv) · [Colab tour](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/overview/ten-minute-tour.ipynb) |

## Contributing

Bug reports, docs, and code: [Contributing](CONTRIBUTING.md). [Discord](https://discord.gg/QPyqFYx2UN).

## License

Apache 2.0.
