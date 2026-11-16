# Modal UDF demo

Each UDF under `[tool.pixeltable.udf]` in `pyproject.toml` runs on its own UDF server in Pixeltable Cloud, with the
CPU, memory and GPU given there. The tables are in the local database; only the UDF calls leave the machine.
`app.py` defines:

- `demo/videos`: uploaded videos, with a 2-second excerpt (`clip()`) and 5-second segments (`segment_video()`).
- `demo/frames`: one frame per second of each video, rotated (`rotate()`), captioned on a T4 GPU
  (`image_captioning()`), with detected objects (`detr_for_object_detection()`) and a CLIP embedding index
  (`clip()`).
- The `demo` service: an insert route, `/videos`, and the query routes `/frames`, `/similar` and `/with-excerpt`.
  `/with-excerpt` appends a video's excerpt to the video with `concat_videos()`.

TODO: remove the /modal-poc folder once the UDF servers are documented in the user docs.

## Setup

You need a Pixeltable Cloud account and [uv](https://docs.astral.sh/uv/). Run all commands in this directory.

1. Pin pixeltable in `pyproject.toml` to the commit of this checkout, which has to be pushed: Pixeltable Cloud builds
   the image of the UDF servers from `uv.lock`, and `pxt` refuses a lock with another pixeltable than the one it
   runs.

   ```bash
   git rev-parse HEAD  # put this into `rev` under [tool.uv.sources], and the repository's URL into `git`
   ```

2. Create the demo's virtual environment:

   ```bash
   uv sync
   source .venv/bin/activate
   ```

   `uv sync` installs pixeltable, and torch and transformers, which the local process needs to load the CLIP model.
   It also writes `uv.lock`.

3. Sign in to Pixeltable Cloud:

   ```bash
   pxt login
   ```

4. In each shell you use for the demo, set:

   ```bash
   export PIXELTABLE_HOME=~/.pixeltable-modal-poc  # the demo's own database, apart from ~/.pixeltable
   export PXT_PORT=22189                           # the demo's own pxt daemon, apart from the default one
   ```

5. Deploy the UDF servers:

   ```bash
   pxt db update
   ```

   Without a database URI, `pxt db update` acts on the local database: it uploads the project, builds an image from
   `uv.lock`, and deploys one UDF server per entry of `[tool.pixeltable.udf]`. The first run took 11 minutes, most of
   it for the image with torch and transformers. `pxt db status` shows the result.

   The `[[tool.pixeltable.database]]` entry named `local` in `pyproject.toml` adds ffmpeg to the image, which the
   video UDFs need.

## Run

Create the tables and start the service:

```bash
pxt schema update app.py demo
pxt service update app.py demo
pxt service list  # shows the service's URL
URL=http://127.0.0.1:<port>
```

Insert two videos from the test data:

```bash
curl -F video_id=1 -F video=@../tests/data/videos/bangkok_half_res.mp4 $URL/videos
curl -F video_id=2 -F "video=@../tests/data/videos/filename with spaces and '_half_res.mp4" $URL/videos
```

The first insert starts the containers of the UDF servers; it took 2 minutes, mostly for the GPU container. A server
stops a container after it has been idle for up to 60 seconds.

Query the tables:

```bash
curl -H 'Content-Type: application/json' -d '{"video_id": 1}' $URL/frames
curl -H 'Content-Type: application/json' -d '{"text": "a cat on a couch"}' $URL/similar
curl -H 'Content-Type: application/json' -d '{"video_id": 1}' $URL/with-excerpt
```

`log_level = "DEBUG"` in `pyproject.toml` makes the service log one line per call to a UDF server:

```bash
grep 'Modal call of' $PIXELTABLE_HOME/logs/*.log
```

To change a UDF's resources, or to add a UDF, edit `[tool.pixeltable.udf]` in `pyproject.toml` and run:

```bash
pxt db update                   # deploys the UDF servers again
pxt service update app.py demo  # restarts the service, which reads the new setting
```

Until the update, a query fails with "The UDF servers of the pixeltable.udf setting run another version of the
project". The same applies to any other change of the project's files.

A UDF that needs a provider's API key reads it from the environment of its server. `pxt secret set pxt://<org>
KEY=value` sets it for the organization, and the next `pxt db update` deploys the UDF servers with it.

## Smoke test

`smoke.py` runs some of the demo's UDFs on their servers, through computed columns and a query, and checks the
results. It creates its table in the database of `$PIXELTABLE_HOME`, whose UDF servers `pxt db update` deployed:

```bash
python smoke.py
```

## Stop

```bash
pxt service stop demo
pxt db delete   # deletes the UDF servers; the local database and its tables stay
pxt daemon stop
```

`pxt daemon stop` leaves the embedded Postgres server of `$PIXELTABLE_HOME` running. This command stops it:

```bash
.venv/lib/python3.*/site-packages/pixeltable_pgserver/pginstall/bin/pg_ctl -D $PIXELTABLE_HOME/pgdata stop
```

Without `pxt db delete`, the UDF servers stay deployed, and run no containers while idle. Pixeltable Cloud deletes
the servers of a local database that was not used for 30 days.
