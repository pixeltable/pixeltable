# Modal UDF demo

PR 1910 contains rudimentary support for running UDFs on Modal. Each UDF under `[tool.pixeltable.udf]` in
`pyproject.toml` runs on its own Modal Server, with the CPU, memory and GPU given there. `app.py` defines:

- `demo/videos`: uploaded videos, with a 2-second excerpt (`clip()`) and 5-second segments (`segment_video()`).
- `demo/frames`: one frame per second of each video, rotated (`rotate()`), captioned on a T4 GPU
  (`image_captioning()`), with detected objects (`detr_for_object_detection()`) and a CLIP embedding index
  (`clip()`).
- The `demo` service: an insert route, `/videos`, and the query routes `/frames`, `/similar` and `/with-excerpt`.
  `/with-excerpt` appends a video's excerpt to the video with `concat_videos()`.

TODO: remove the /modal-poc folder once we have full support for running UDFs on Modal (for local and hosted tables).

## Setup

You need a Modal account and [uv](https://docs.astral.sh/uv/). The service runs `uv export` to build the Modal image,
so uv has to be on the PATH of the shell that runs `pxt`. Run all commands in this directory.

1. Create the demo's virtual environment:

   ```bash
   uv sync
   source .venv/bin/activate
   ```

   `uv sync` installs pixeltable from this checkout, and torch and transformers, which the local process needs to
   load the CLIP model. It also writes `uv.lock`: the Modal image installs the packages listed there, and gets
   pixeltable from this checkout.

2. Sign in to Modal, which writes your token to `~/.modal.toml`:

   ```bash
   modal token new
   ```

3. Create a proxy token, with a key that starts with `wk-` and a secret that starts with `ws-`. The Servers accept
   only requests with that token.

   ```bash
   modal workspace proxy-tokens create
   ```

4. In each shell you use for the demo, set:

   ```bash
   export PIXELTABLE_HOME=~/.pixeltable-modal-poc  # the demo's own database, apart from ~/.pixeltable
   export PXT_PORT=22189                           # the demo's own pxt daemon, apart from the default one
   export MODAL_PROXY_KEY=wk-...                   # the proxy auth token
   export MODAL_PROXY_SECRET=ws-...
   ```

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

The first insert builds the image of the Modal App `pxt-udf-modal-poc-<hash>`, deploys the App and starts the
containers; it took 3 minutes. The second insert took 1 minute, while the containers were still running. Modal stops a container
after it has been idle for up to 60 seconds, and starting them again takes about 2 minutes, mostly for the GPU
container.

Query the tables:

```bash
curl -H 'Content-Type: application/json' -d '{"video_id": 1}' $URL/frames
curl -H 'Content-Type: application/json' -d '{"text": "a cat on a couch"}' $URL/similar
curl -H 'Content-Type: application/json' -d '{"video_id": 1}' $URL/with-excerpt
```

`log_level = "DEBUG"` in `pyproject.toml` makes the service log one line per Modal call:

```bash
grep 'Modal call of' $PIXELTABLE_HOME/logs/*.log
```

To change a UDF's resources, edit its entry under `[tool.pixeltable.udf]` in `pyproject.toml` and run:

```bash
pxt daemon restart              # the daemon reads the configuration when it starts
pxt service update app.py demo  # restarts the service, which deploys the App again
```

## Smoke test

`smoke.py` runs some of the demo's UDFs on Modal, through computed columns and a query, and checks the results. It
creates its tables in a new Pixeltable home; the last command stops that home's Postgres server:

```bash
SMOKE_HOME=$(mktemp -d)
PIXELTABLE_HOME=$SMOKE_HOME python smoke.py
.venv/lib/python3.*/site-packages/pixeltable_pgserver/pginstall/bin/pg_ctl -D $SMOKE_HOME/pgdata stop
```

At its first Modal call, the script deploys the App again, which stops the containers of a running demo service.

## Stop

```bash
pxt service stop demo
pxt daemon stop
```

`pxt daemon stop` leaves the embedded Postgres server of `$PIXELTABLE_HOME` running. This command stops it:

```bash
.venv/lib/python3.*/site-packages/pixeltable_pgserver/pginstall/bin/pg_ctl -D $PIXELTABLE_HOME/pgdata stop
```

The Modal App stays deployed, but its Servers run no containers while idle. `modal app stop pxt-udf-modal-poc-<hash>`
removes it.
