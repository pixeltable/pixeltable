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
[**Discord**](https://discord.gg/QPyqFYx2UN) |
[**For agents**](https://www.pixeltable.com/get-started.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
[![tests status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml)
[![nightly status](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml)
[![stress-tests status](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml)
[![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)
[![Python](https://img.shields.io/pypi/pyversions/pixeltable)](https://pypi.org/project/pixeltable/)

Install into your agent: [Agent Skill](https://github.com/pixeltable/pixeltable-skill) ·
[MCP server](https://github.com/pixeltable/mcp-server-pixeltable-developer) ·
[Cursor plugin](https://cursor.directory/plugins/pixeltable) ·
[Claude plugin](https://claude.com/marketplace/plugins/pixeltable) ·
[ChatGPT plugin](https://chatgpt.com/plugins/plugins_6aa9bce6481c81918d73be74740eed28)

Build AI apps on video, images, audio and documents in seconds. A table holds references to your files next to
regular columns: a path or URL you insert stays where it is, and files you upload or Pixeltable produces go to its
media store, on local disk or in [your own S3, GCS or Azure bucket](https://docs.pixeltable.com/integrations/cloud-storage). The work you want done on them, like
pulling the audio out of a video, transcribing it or embedding a summary, is a column too. Insert a row and
Pixeltable runs that work, stores the results and indexes them, with no embed-and-upsert job to run. Change a
prompt later and only the summaries rerun; the transcripts you already paid for stay.

That replaces the stack most of us end up with: a bucket for the files, a database for the metadata, a vector
database for search, an orchestrator for the model calls, an API on top, and the scripts that keep it all in sync.
Here it's one file, `app.py`.

Keep the database your app already runs for users and auth. Pixeltable handles the media and model work next to
Supabase, Convex or a managed Postgres.

## What it looks like

A small video API: upload a video with an audio track, get a summary back, then search across everything you
uploaded. This is all of `app.py`:

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
# the first pxt command starts a local database daemon, which reads your environment once
export OPENAI_API_KEY=sk-...        # set it first, or run pxt daemon restart after
pxt init                            # makes this folder a project
pxt schema update app.py my_app     # creates the table and its index
pxt service update app.py my_app    # starts the service and prints its URL
```

```bash
# PORT is the one pxt service update printed
curl -F video=@talk.mp4 http://127.0.0.1:PORT/videos
# {"summary": "..."}
curl -X POST http://127.0.0.1:PORT/search -H 'Content-Type: application/json' -d '{"query": "pricing"}'
# {"rows": [{"video": "...", "summary": "..."}]}
```

Each column you compute says which columns it reads, so Pixeltable knows what to run and in what order. In practice:

- Upload 100 more videos and only those 100 get processed.
- Insert 10,000 videos and the process doing the insert runs the model calls: the service for a route, your Python
  process for a script. It keeps many requests in flight and stays under the provider's rate limits, read from the
  provider's responses where it reports them or set in `config.toml` (600 requests a minute by default). On
  Pixeltable Cloud you set CPU, memory and worker counts in the project config.
- Change the summary prompt, run `pxt schema update` and `pxt service update` again, and new uploads use it.
  `pxt recompute my_app/videos summary` then redoes the old summaries without touching the transcripts you already
  paid for.
- Bulk-load from Python with `on_error='ignore'` and a failed model call leaves its error on that row instead of
  stopping the batch. `pxt recompute my_app/videos transcript --errors-only` reruns just those rows.
- Every change to a table is versioned. `pxt history` shows it, `pxt revert` undoes it.

[How it works](https://docs.pixeltable.com/overview/how-it-works) explains the rules behind each of these.

## Why we built it

We kept seeing that same stack in every multimodal AI project, and each piece is fine on its own. The trouble is in
between: a sync script that hopes the bucket and the database still match, and a vector database still serving
embeddings of last week's summaries. Most of the code was plumbing, and so were most of the bugs.

We're database people (Pixeltable comes from the creators of Apache Parquet and Impala), so our answer was to move
the work into the database. The engine tracks what depends on what, so recomputing the right rows is its job, not
yours.

It won't make your models better. Media is still big, model calls still fail and evaluation is still on you. What
goes away is the plumbing you rebuild every time you want to try a better prompt. The longer version is on
[Why Pixeltable](https://docs.pixeltable.com/overview/pixeltable).

## How it's different

| You need to | Usually | With Pixeltable |
|---|---|---|
| Store video, images, audio, PDFs | A bucket plus a metadata table | A column of type video, image, audio or document; the row holds a reference to the file and its results (`pxt.Video`, `pxt.Image`, `pxt.Audio`, `pxt.Document`) |
| Call models on them | An orchestrator or a queue, plus scripts | A column whose value is a model call, run on every new row and stored (a computed column) |
| Search them | A vector database you keep in sync | An index on that column, built and updated by the table (`pxt.EmbeddingIndex`) |
| Split documents, sample frames | A preprocessing job | A table derived from the first, one row per chunk or frame, updated as rows come and go (a view) |
| Serve the result | A FastAPI app you write | Routes declared next to the tables (`FastAPIRouter`) |
| Undo a bad change | Restore a backup | Go back to an earlier version of the table (`pxt revert`) |

If you don't want a schema, Pixeltable is the wrong tool. Coming from Pinecone, Postgres, LangGraph or Airflow?
[Migrate](https://docs.pixeltable.com/howto/coming-from) shows what each one becomes.

## What you can build

- [RAG over documents, with no re-chunk or re-embed scripts when you add or remove documents](https://docs.pixeltable.com/use-cases/multimodal-backend)
- [Video and image pipelines: frames, transcripts, detections](https://docs.pixeltable.com/use-cases/media-processing)
- [Agents whose tool calls are rows you can query later](https://docs.pixeltable.com/use-cases/agentic-workflows)

Rather not start from a blank file?

```bash
pxt service example --out app.py   # a working app: a table, computed columns and HTTP routes
pxt schema example                 # prints every construct a schema can use
```

For complete apps, browse
[awesome-pixeltable-apps](https://github.com/pierrebrunelle/awesome-pixeltable-apps): each one opens in GitHub
Codespaces.

## Install

```bash
pip install 'pixeltable[serve]'
```

Python 3.11 or later. The [quickstart](https://docs.pixeltable.com/overview/quick-start) goes from an empty folder
to your first POST. Rather start in a notebook? The
[10-minute tour](https://docs.pixeltable.com/overview/ten-minute-tour) needs no CLI.
[HTTP serving](https://docs.pixeltable.com/howto/deployment/serving) covers uploads and calling your routes from
TypeScript. [Self-hosting](https://docs.pixeltable.com/howto/deployment/overview) covers your own FastAPI app and
running without HTTP.

**Pixeltable Cloud** is for running this in production: the same `app.py` on a hosted database, with media in a
managed bucket and CPU, memory and worker counts set in the project config. The
[Cloud guide](https://docs.pixeltable.com/cloud) goes from `pxt login` to a running service. To try it with no
account, `pxt new` gives you a free Cloud database for about 48 hours and prints a link to keep it.

## Using a coding agent?

```bash
npx skills add pixeltable/pixeltable-skill
# or, from the root of your repo, with only Python:
pxt skills install
```

The skill teaches your agent to write `app.py` the way it's written above. One file helps here: the agent writes the
whole backend, and you review one diff instead of a change in every system.
[AI coding agents](https://docs.pixeltable.com/overview/building-pixeltable-with-llms) covers setup for each tool,
the plugins and the MCP servers.

## Contributing

Bug reports and PRs are welcome: [CONTRIBUTING.md](https://github.com/pixeltable/pixeltable/blob/main/CONTRIBUTING.md).
Questions go to [Discord](https://discord.gg/QPyqFYx2UN) or
[GitHub Discussions](https://github.com/pixeltable/pixeltable/discussions). Pixeltable is Apache 2.0.
