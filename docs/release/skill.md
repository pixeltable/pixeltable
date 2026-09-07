---
name: pixeltable
description: >
  Build multimodal AI applications with Pixeltable. One application file declares
  TableModel tables, computed columns, embedding indexes, and FastAPI routes; inserting
  a row runs the transforms. Use when building RAG, processing images, video, audio, or
  documents, or serving an API over that data. Do not use for general Python or direct
  PostgreSQL administration.
license: Apache-2.0
compatibility: Requires Python 3.11+ on Linux, macOS, or Windows. HTTP serving needs the `serve` extra.
metadata:
  author: Pixeltable
  version: "1.0"
---

# Pixeltable

Pixeltable is the database, the orchestration, and the serving in one Python file. Tables
store the data, computed columns declare the transforms, embedding indexes make it
searchable, and `FastAPIRouter` exposes it over HTTP. Insert a row and the transforms run.

## Install and first run

```bash
pip install -U 'pixeltable[serve]'
pxt init                              # mark this directory a project root
pxt service example --out app.py      # write a working application file
pxt schema update app.py my_app       # create the tables the models declare
pxt service update app.py my_app      # serve this file's routes
pxt service list                      # print the assigned URL
```

`pxt init` is a prerequisite: `pxt schema update` refuses a file that sits under no project
root. `schema update` creates tables and does not start endpoints; `service update` starts
endpoints and does not create tables. Both prompt for confirmation unless you pass `-f`,
and exit 3 when run non-interactively without it.

The last argument (`my_app`) names a catalog inside Pixeltable. It is not a folder on disk.

## The application file

```python
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


@pxt.udf
def excerpt(text: str, n: int = 12) -> str:
    return text if len(text) <= n else f'{text[:n]}...'


class Docs(TableModel, name='docs'):
    doc_id: pxt.Int
    title: pxt.String
    body: pxt.String | None
    title_upper = pxtf.string.upper(title)   # computed: an assignment, not an annotation
    summary = excerpt(title)


ingest = FastAPIRouter(name='ingest')
ingest.add_insert_route(
    Docs, path='/docs', inputs=[Docs.doc_id, Docs.title, Docs.body], outputs=[Docs.title_upper, Docs.summary]
)
ingest.add_compute_route(Docs, path='/titles', inputs=[Docs.title], outputs=[Docs.title_upper])
```

An **annotation** (`title: pxt.String`) is a value you insert. An **assignment**
(`title_upper = ...`) is a computed column, recomputed on insert and on update. A
non-nullable annotated column that is a route input is a required field in the request
body:

```bash
curl -X POST http://127.0.0.1:<port>/docs \
  -H 'Content-Type: application/json' \
  -d '{"doc_id": 1, "title": "Hello", "body": "world"}'
# {"title_upper":"HELLO","summary":"Hello"}
```

## Capabilities

- **Multimodal columns**: `pxt.Image`, `pxt.Video`, `pxt.Audio`, `pxt.Document`, plus
  `pxt.String`, `pxt.Int`, `pxt.Float`, `pxt.Bool`, `pxt.Json`, `pxt.Array`, timestamps.
- **Computed columns** call any UDF or provider function and run incrementally: only new
  or changed rows compute.
- **Views with iterators** expand one row into many. `frame_iterator` for video,
  `document_splitter` for documents, `audio_splitter`, `string_splitter`, `tile_iterator`.
- **Embedding indexes** declared in `__indexes__`; query with
  `column.similarity(string=...)` or `similarity(image=...)`.
- **UDFs** with `@pxt.udf`, aggregates with `@pxt.uda`, reusable queries with `@pxt.query`.
- **Serving**: `add_insert_route`, `add_compute_route`, `add_update_route`,
  `add_delete_route`, `add_query_route`. `FastAPIRouter` subclasses
  `fastapi.APIRouter`, so `app.include_router(...)` mounts it on an existing app.
- **Providers**: OpenAI, Anthropic, Gemini, Bedrock, Mistral, Together, Fireworks, Groq,
  Replicate, Hugging Face, Ollama, vLLM, Voyage, Jina, and more under
  `pixeltable.functions`.

## Constraints

- Application code declares a `TableModel` in `app.py` and creates it with
  `pxt schema update`. It does **not** call `pxt.create_table()` or
  `add_embedding_index()`; indexes belong in `__indexes__`.
- Notebooks, tests, and the REPL do use `pxt.create_table()` and
  `add_embedding_index()`. That is correct there and does not need a project file.
- Importing `app.py` declares the models but does not attach them to tables. Call
  `TableModel.bind_all('<target>')` before inserting or querying from plain Python.
- A UDF is referenced by the file path it is defined in. Moving or renaming that file
  leaves the columns that call it unable to compute.
- `pxt service run` always serves from the current process and cannot target Cloud.

## Do not reach for

Chunking, retrieval, tool-calling, and orchestration are built in. Adding these fights
the model rather than helping it:

- LangChain, LlamaIndex, or Haystack for chunking, retrieval, or tool-calling
- A separate vector database; embedding indexes live on the table
- pandas as a working store; the table is the store
- A per-row `for` loop calling a model; use a computed column
- A manual agent `while` loop; model the agent as a table

## Cloud

Pixeltable Cloud is in Limited Beta. Email contact@pixeltable.com if you are interested.
The same application file targets a hosted database with `pxt db update`,
`pxt schema update`, and `pxt service update` against a `pxt://org:db` target, once
`PIXELTABLE_API_KEY` is set.

## Reference

- Documentation: https://docs.pixeltable.com/
- Quickstart: https://docs.pixeltable.com/overview/quick-start
- SDK reference: https://docs.pixeltable.com/sdk/latest/pixeltable
- Coding-agent skill: `npx skills add pixeltable/pixeltable-skill`
