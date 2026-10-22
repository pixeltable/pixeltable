<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/release/_logo/pxt-dark.svg">
  <img alt="Pixeltable" src="https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/release/_logo/pxt-light.svg" width="40%">
</picture>

**Build AI apps on video, images, audio and documents in one Python file.**

[Quickstart](https://docs.pixeltable.com/overview/quick-start) ·
[Docs](https://docs.pixeltable.com/) ·
[Discord](https://discord.gg/QPyqFYx2UN)

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)
[![Python](https://img.shields.io/pypi/pyversions/pixeltable)](https://pypi.org/project/pixeltable/)
[![tests status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml)

Pixeltable is an open-source Python framework. Your files live in tables, next to regular columns. The work you
want done on them, like pulling the audio out of a video, transcribing it or embedding a summary, is a column too.
Insert a row and Pixeltable runs that work, stores the results and keeps the search index up to date.

That replaces the stack most of us end up with: a bucket for the files, a database for the metadata, a vector
database for search, an orchestrator for the model calls, an API on top, and the scripts that keep it all in sync.
Here it's one file, `app.py`.

## What it looks like

A small video API. Upload a video, get a summary back, then search across everything you uploaded.

```python
import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions.video import extract_audio
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()
embed = openai.embeddings.using(model='text-embedding-3-small')


class Videos(TableModel, name='videos'):
    video: pxt.Video  # the only column you insert; the rest run on insert

    audio = extract_audio(video, format='mp3')
    transcript = openai.transcriptions(audio, model='whisper-1')
    summary = openai.chat_completions(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': 'Summarize this video in two sentences.'},
            {'role': 'user', 'content': transcript.text},
        ],
    ).choices[0].message.content.astype(pxt.String)

    __indexes__ = [pxt.EmbeddingIndex(summary, embedding=embed)]


@pxt.query
def search(query: str):
    sim = Videos.summary.similarity(string=query)
    return Videos.order_by(sim, asc=False).limit(3).select(Videos.video, Videos.summary)


api = FastAPIRouter(name='video-api')
api.add_insert_route(Videos, path='/videos', uploadfile_inputs=[Videos.video], outputs=[Videos.summary])
api.add_query_route(path='/search', query=search)
```

```bash
pip install 'pixeltable[serve]' openai
export OPENAI_API_KEY=sk-...
pxt init                            # writes pixeltable.toml
pxt schema update app.py my_app     # creates the table and its index
pxt service update app.py my_app    # starts the API and prints its URL
```

```bash
curl -F video=@talk.mp4 http://127.0.0.1:PORT/videos
# {"summary": "..."}
curl -X POST http://127.0.0.1:PORT/search -H 'Content-Type: application/json' -d '{"query": "pricing"}'
# {"rows": [{"video": "...", "summary": "..."}]}
```

Each column says what it reads from, so Pixeltable knows what to run and in what order. In practice:

- Upload 100 more videos and only those 100 get processed.
- Change the summary prompt and new uploads use it. `pxt recompute my_app/videos summary` redoes the old
  summaries without touching the transcripts you already paid for.
- Bulk-load from Python with `on_error='ignore'` and a failed model call leaves its error on that row instead of
  stopping the batch. `pxt recompute my_app/videos transcript --errors-only` reruns just those rows.
- Every change to a table is versioned. `pxt history` shows it, `pxt revert` undoes it.

## Why we built it

We kept seeing that same stack in every multimodal AI project, and each piece is fine on its own. The trouble is in
between: a sync script that hopes the bucket and the database still match, an orchestrator that reruns everything
because it can't tell what changed. Most of the code was plumbing, and so were most of the bugs.

We're database people (Pixeltable comes from the creators of Apache Parquet and Impala), so our answer was to move
the work into the database. A transform is a column. The engine tracks what depends on what, so recomputing the
right rows is its job, not yours.

It won't make your models better. Media is still big, model calls still fail and evaluation is still on you. What
goes away is the plumbing you rebuild every time you want to try a better prompt.

## How it's different

| You need to | Usually | With Pixeltable |
|---|---|---|
| Store video, images, audio, PDFs | A bucket plus a metadata table | `pxt.Video`, `pxt.Image`, `pxt.Audio`, `pxt.Document` columns |
| Call models on them | An orchestrator or a queue, plus scripts | Computed columns that run on insert |
| Search them | A vector database you keep in sync | An embedding index on a column |
| Split documents, sample frames | A preprocessing job | A view with one row per chunk or frame |
| Serve the result | A FastAPI app you write | `FastAPIRouter` routes in the same file |
| Undo a bad change | Restore a backup | `pxt revert` |

If you don't want a schema, Pixeltable is the wrong tool. Coming from Pinecone, Postgres, LangGraph or Airflow?
[Migrate](https://docs.pixeltable.com/howto/coming-from) shows what each one becomes.

## What you can build

- [RAG that stays in sync with your documents](https://docs.pixeltable.com/use-cases/multimodal-backend)
- [Video and image pipelines: frames, transcripts, detections](https://docs.pixeltable.com/use-cases/media-processing)
- [Agents whose tool calls are rows you can query later](https://docs.pixeltable.com/use-cases/agentic-workflows)

To start from a full app, `uvx pixeltable-new myapp` copies a chat agent over your documents, and
`uvx pixeltable-new myapp --video` copies video search. Both live in the
[starter kit](https://github.com/pixeltable/pixeltable-starter-kit).

## Install

```bash
pip install 'pixeltable[serve]'
```

Python 3.11 or later. The [quickstart](https://docs.pixeltable.com/overview/quick-start) goes from an empty folder
to your first POST. Rather start in a notebook? The
[10-minute tour](https://docs.pixeltable.com/overview/ten-minute-tour) needs no CLI.
[Self-hosting](https://docs.pixeltable.com/howto/deployment/overview) covers Docker, your own FastAPI app and running
without HTTP.

**Pixeltable Cloud** runs the same `app.py` on a hosted database. It's in Limited Beta: email
[contact@pixeltable.com](mailto:contact@pixeltable.com) if you'd like access. Setup is in the
[Cloud guide](https://docs.pixeltable.com/cloud).

## Using a coding agent?

```bash
npx skills add pixeltable/pixeltable-skill
```

The skill teaches your agent to write `app.py` the way it's written above. One file helps here: the agent writes the
whole backend, and you review one diff instead of changes across four systems. There are also plugins for
[Claude](https://claude.com/marketplace/plugins/pixeltable), [Cursor](https://cursor.directory/plugins/pixeltable)
and [ChatGPT](https://chatgpt.com/plugins/plugins_6aa9bce6481c81918d73be74740eed28), and
[MCP servers](https://docs.pixeltable.com/overview/building-pixeltable-with-llms#choose-a-pixeltable-mcp-server)
for the docs, your local catalog and your Cloud org.

If you're an agent reading this, start at [get-started.md](https://www.pixeltable.com/get-started.md).

## Contributing

Bug reports and PRs are welcome: [CONTRIBUTING.md](https://github.com/pixeltable/pixeltable/blob/main/CONTRIBUTING.md).
Questions go to [Discord](https://discord.gg/QPyqFYx2UN) or
[GitHub Discussions](https://github.com/pixeltable/pixeltable/discussions). Pixeltable is Apache 2.0.
