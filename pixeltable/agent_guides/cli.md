# CLI

Running `pxt` in a project: the project root, the daemon, application files, prompts, exit codes and catalog inspection.

## The project root

- `pxt init` makes the working directory the project root: it writes `pixeltable.toml` with a `[[pixeltable.database]]`
  entry for the local database. In a directory with a `pyproject.toml`, it appends `[[tool.pixeltable.database]]` to
  that file instead.
- In a directory that is already a project root, `pxt init` prints the name of the project configuration file and
  changes nothing.
- `pxt init` exits 3 in the home directory, in a directory above it, in `$PIXELTABLE_HOME` (default `~/.pixeltable`),
  or under another project root.
- `pxt` searches up from the working directory for the project root: the first directory with a `pixeltable.toml`, or
  with a `pyproject.toml` that has a `[tool.pixeltable]` section.
- Run each command for a project from its root or below it.
- `pxt schema` and `pxt service` refuse an application file outside a project root.
- The catalog records a UDF by its module path relative to the root (`app.excerpt` for `excerpt()` in `app.py`), so
  each directory between the root and the file needs a Python identifier as its name.

## The daemon

- The daemon is the background process that serves `pxt` commands.
- If the daemon is not running, a command that needs it starts one on `127.0.0.1:22089`, or on the port in `PXT_PORT`.
- Later commands send their requests to the same daemon.
- The daemon serves one project. A command run under another project root restarts the daemon for that project; a
  command run outside any project uses the running daemon.
- A command also restarts the daemon when the Pixeltable version, the Python interpreter, the installed packages or a
  `PIXELTABLE_*` variable differs from the daemon's.
- The daemon reads its other settings once, at startup: environment variables such as `OPENAI_API_KEY`, and the values
  in `$PIXELTABLE_HOME/config.toml`.
- After one of them changes, the next command that reads or changes table data restarts the daemon with the new
  values, then runs.
- `pxt config`, `pxt ls` and `pxt describe` leave the daemon as it is; `pxt config` lists the variables whose values
  differ between the shell and the daemon.
- While the daemon serves a request, a command that has to restart it fails with `DAEMON_BUSY`, which is retryable.
- `pxt daemon restart` refuses while the daemon serves a request; with `-f` it restarts anyway and stops every request
  in progress.
- The daemon starts each local service as a separate process with the daemon's environment.
- A service keeps that environment across `pxt daemon restart`; `pxt service restart my_app/ingest` restarts it with
  the new environment.

## Applying an application file

- `pxt schema check app.py` imports the file without reading the catalog, and checks that another process can import
  each UDF called by its columns.
- `pxt schema check` warns about a project module that has the name of an installed package.
- `pxt schema check` also warns about a computed column that references no column, such as `label = str(title)`: such
  a column stores the same value in every row.
- `pxt functions` lists functions and needs no daemon or catalog (`pxt guide expressions`).
- `pxt schema diff app.py my_app` prints the planned changes to the tables under `my_app`.
- `pxt schema update app.py my_app` applies these changes and starts no service; `pxt guide tables` says what each
  kind of change does.
- `pxt service update app.py my_app` starts each service of the file in a background process, on a port chosen by the
  OS.
- `pxt service update` restarts a running service when its routes, the project's files or `--otel` have changed.
- A restarted service keeps its port.
- `pxt service update` fails while the tables differ from the models; it exits 1 with a
  `next: pxt schema update ...` line.
- `pxt service list` prints each service's endpoint, process id and routes; with `--json`, each service is an object
  with its `endpoint` and `port`.
- `pxt service logs` reads the logs of hosted services only.
- For a local service, `pxt service logs` exits 1, and its error gives the path of the log file, such as
  `$PIXELTABLE_HOME/logs/services/my_app/ingest.log` for `my_app/ingest`.
- `pxt service update` leaves a service running when the file no longer defines it; `pxt service prune app.py my_app`
  stops it.
- `pxt service run app.py my_app` serves one service in the foreground and records nothing, so `pxt service list` and
  `pxt service stop` do not see it.
- `pxt recompute my_app/docs title_upper` recomputes stored values; `pxt guide tables` says which rows and columns it
  recomputes.

## Confirmation and exit codes

- `pxt schema`, `pxt service` and `pxt db` exit 0 when done or when nothing is pending, and exit 1 on an error.
- A `diff`, or an `update` or `prune` with `-n`, exits 2 when changes are pending.
- Every command exits 3 on a refusal.
- These commands ask for confirmation: `pxt service update` when it restarts a running service or makes a destructive
  change; `pxt schema update` with a destructive change; the `prune` subcommands, `pxt drop`, `pxt drop-dir`,
  `pxt recompute`, `pxt revert`, `pxt db update` and `pxt db delete`.
- `-f` skips the confirmation.
- An additive `pxt schema update` runs without asking, and so does a `pxt service update` that only starts services,
  such as the first one.
- Without a terminal, a command that would ask exits 3 and prints a `next:` line on stderr with the command to rerun:
  `next: pxt service update app.py my_app -f  # start or restart 1 service(s), interrupting 1 running service(s)`.
- `pxt schema update` refuses to drop a column or an index, and `pxt service update` refuses to change or remove a
  route, unless given `--allow-destructive`.
- On that refusal, the command exits 3 and applies nothing, not even the additive changes; its `next:` line repeats
  the command with `--allow-destructive`.
- On a command that changes something, `-n` (`--dry-run`) prints the planned changes and applies nothing.
- With `--json`, the output is one JSON document on stdout.
- On a failure, the document also has `error_code`, `message` and `retryable`, and `retry_after` and `next` when they
  apply.
- `next` is a list of `{"command": ..., "effect": ...}`.
- The error message also goes to stderr.

## Inspecting the catalog

- A catalog path separates directories with `/`: `my_app/docs`. `pxt count my_app.docs` exits 2 with
  `pxt paths use '/' as the separator`.
- `pxt rows my_app/docs` prints the first 10 rows.
- `pxt rows` leaves out unstored computed columns unless `--cols` lists them, since it would have to compute them.
- `pxt get my_app/docs d1` looks up a row by primary key.
- `pxt errors my_app/docs` lists the rows that have an error in a computed column.
- `pxt get` and `pxt errors` need a table with a primary key.
- `pxt errors` ends with a `next:` line for each column whose failed cells can be recomputed.
- `pxt config` prints every setting with its value and its source: an environment variable, a file, or `unset`.
  Credentials are redacted.
- `pxt feedback --kind gap 'TEXT'` prints a report for the Pixeltable team without sending it; `--send` sends it.
- With `DO_NOT_TRACK` set to `1`, `true` or `yes`, `--send` sends nothing and exits 3.
- `pxt dashboard` opens the daemon's dashboard in a browser and prints its URL.

```bash
pxt init
pxt schema check app.py
pxt schema diff app.py my_app
pxt schema update app.py my_app
pxt service update app.py my_app -f
pxt service list my_app --json
pxt daemon restart
pxt service restart my_app/ingest
pxt service prune app.py my_app -n
pxt ls my_app
pxt describe my_app/docs
pxt rows my_app/docs -n 5 --cols id,title
pxt get my_app/docs d1
pxt count my_app/docs
pxt errors my_app/docs --col title_upper
pxt history my_app/docs -n 5
pxt config --section openai
pxt dashboard
```
