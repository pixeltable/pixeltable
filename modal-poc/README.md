# Modal UDF demo

Each UDF under `[tool.pixeltable.udf]` in `pyproject.toml` runs on its own Modal Server, with the CPU, memory and GPU
given there. The tables are in the local database; only the calls to those UDFs leave the machine. `app.py` defines:

- `demo/videos`: uploaded videos, with a 2-second excerpt (`clip()`) and 5-second segments (`segment_video()`).
- `demo/frames`: one frame per second of each video, rotated (`rotate()`), captioned on a T4 GPU
  (`image_captioning()`), with objects detected on a T4 GPU (`detr_for_object_detection()`) and drawn locally
  (`bboxes_draw()`), and a CLIP embedding index (`clip()`).
- The `demo` service: an insert route, `/videos`, and the query routes `/frames`, `/similar` and `/with-excerpt`.
  `/with-excerpt` appends a video's excerpt to the video with `concat_videos()`.

TODO: remove the /modal-poc folder once the UDF servers are fully rolled out.

## Setup

You need [uv](https://docs.astral.sh/uv/), and a Pixeltable Cloud account or a Modal account. Run all commands in this
directory.

1. Point `pixeltable` under `[tool.uv.sources]` in `pyproject.toml` at this checkout:

   - With Pixeltable Cloud, pin the commit, which has to be pushed: Pixeltable Cloud builds the image of the UDF
     servers from `uv.lock`, and `pxt` refuses a lock with another `pixeltable` than the one it runs.

     ```bash
     git rev-parse HEAD  # put this into `rev`, and the repository's URL into `git`
     ```

   - With your own Modal workspace, use the checkout itself: `pixeltable = { path = "..", editable = true }`.
     `deploy_modal_servers.py` adds it to the image of the UDF servers.

2. Create the demo's virtual environment. With your own Modal workspace, add `--extra modal-cli --python 3.12` to
   `uv sync`: the image of the UDF servers runs the local Python version, and Modal supports 3.10 to 3.13.

   ```bash
   uv sync
   source .venv/bin/activate
   ```

   `uv sync` installs `pixeltable`, and `torch` and `transformers`, which the local process needs to load the CLIP model
   for text search queries. It also writes `uv.lock`.

3. In each shell you use for the demo, set:

   ```bash
   export PIXELTABLE_HOME=~/.pixeltable-modal-poc  # the demo's own database, apart from ~/.pixeltable
   export PXT_PORT=22189                           # the demo's own pxt daemon, apart from the default one
   ```

4. Deploy the UDF servers with Pixeltable Cloud:

   ```bash
   pxt login
   pxt db update
   ```

   Without a database URI, `pxt db update` acts on the local database: it uploads the project, builds an image from
   `uv.lock`, and deploys one UDF server per entry of `[tool.pixeltable.udf]`. `pxt db status` shows the result. The
   `[[tool.pixeltable.database]]` entry named `local` in `pyproject.toml` adds `ffmpeg` to the image, which the video
   UDFs need.

   Or deploy them to your own Modal workspace:

   ```bash
   modal token new  # once
   python deploy_modal_servers.py
   source .modal_env.sh
   ```

   `deploy_modal_servers.py` deploys the Modal App `pxt-udf-modal-poc`, and `.modal_env.sh` sets the
   `PXT_UDF_SERVERS` and `PXT_UDF_SERVERS_HEADERS` that send the UDF calls to it.

## Run

`run_demo.py` runs this section in a database of its own, `.pxt_home`, saves an annotated frame of each video to
`output_media/`, prints the Modal calls per UDF, and stops the service and the daemon:

```bash
python run_demo.py
```

By hand, create the tables and start the service:

```bash
pxt schema update app.py demo
pxt service update app.py demo -f
pxt service list  # shows the service's URL
URL=http://127.0.0.1:<port>
```

Insert two videos from the test data:

```bash
curl -F video_id=1 -F video=@../tests/data/videos/bangkok_half_res.mp4 $URL/videos
curl -F video_id=2 -F "video=@../tests/data/videos/filename with spaces and '_half_res.mp4" $URL/videos
```

The first insert starts the containers of the UDF servers and caches Hugging Face weights in the shared `pxt-hf-cache`
Modal volume. A server stops a container after it has been idle for up to 60 seconds.

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
pxt db update                   # or: python deploy_modal_servers.py, then source .modal_env.sh
pxt service update app.py demo  # restarts the service, which reads the new setting
```

Until the update, a query fails with "The UDF servers of the pixeltable.udf setting run another version of the
project". The same applies to any other change of the project's files.

A UDF that needs a provider's API key reads it from the environment of its server. `pxt secret set pxt://<org>
KEY=value` sets it for the organization, and the next `pxt db update` deploys the UDF servers with it.
`pxt db update` refuses a project configuration file with a secret in it, such as an API key.

## Smoke test

`smoke.py` runs some of the demo's UDFs on their servers, through computed columns and a query, and checks the
results. It creates its table in the database of `$PIXELTABLE_HOME`, whose UDF servers you deployed above:

```bash
python smoke.py
```

## Stop

```bash
pxt service stop demo
pxt db delete   # deletes the UDF servers; the local database and its tables stay
pxt daemon stop
```

With your own Modal workspace, `modal app stop pxt-udf-modal-poc` replaces `pxt db delete`.

`pxt daemon stop` leaves the embedded Postgres server of `$PIXELTABLE_HOME` running. This command stops it:

```bash
.venv/lib/python3.*/site-packages/pixeltable_pgserver/pginstall/bin/pg_ctl -D $PIXELTABLE_HOME/pgdata stop
```

Without `pxt db delete`, the UDF servers stay deployed, and run no containers while idle. Pixeltable Cloud deletes
the servers of a local database that was not used for 30 days.
