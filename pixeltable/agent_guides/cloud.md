# Cloud

Deploying an application file to Pixeltable Cloud with `pxt`: sign-in, hosted databases, secrets and logs.

## Signing in

- `pxt login` prints a code and a URL, and returns once the user confirms the code in a browser, on this machine or
  another.
- `pxt login` waits up to 10 minutes for the confirmation.
- `pxt login` caches the session in `$PIXELTABLE_HOME` (default `~/.pixeltable`); when the session token expires, the
  next command renews it.
- An API key takes precedence over a session.
- Pixeltable reads the API key from `PIXELTABLE_API_KEY`, or from `api_key` in the `[pixeltable]` section of
  `$PIXELTABLE_HOME/config.toml`.
- A change to `PIXELTABLE_API_KEY` restarts the daemon at the next command (`pxt guide cli`).
- `pxt whoami` shows the credential in use, and whether Pixeltable Cloud recognizes it.
- Without a credential, a command that calls Pixeltable Cloud fails with `MISSING_CREDENTIALS`.

## The database entry

A hosted database has the URI `pxt://ORG:DB`, such as `pxt://acme:main`. The organization and database names use
lowercase letters, digits and hyphens, start with a letter, end with a letter or digit, and have at most 29
characters. The project configuration describes the database in a separate `[[pixeltable.database]]` entry, after the
local entry from `pxt init`:

```toml
[[pixeltable.database]]

[[pixeltable.database]]
name = 'pxt://acme:main'
cpu = 2
memory_mb = 4096
workers = 1

[pixeltable.database.services.'my_app/ingest']
workers = 2
file_cache_gb = 5
```

- `pxt db update pxt://acme:main` uses the entry whose `name` is the URI. Without one, it fails and gives the command
  that appends it: `pxt db diff pxt://acme:main --add-config-entry`.
- `--add-config-entry` on `pxt db diff` or `pxt db update` appends the entry, with `name = 'pxt://acme:main'`, unless
  one with that name exists.
- The entry's keys also select the uploaded project files and the contents of the database's image, and set the
  database's capacity; `pxt db --help` describes the keys.
- An unknown key, such as `secrets`, makes the configuration fail to load.
- `services` sets the capacity of each hosted service; each key is `'<directory>/<service name>'`, or the service name
  alone for a service at the database root.
- `pxt service update` applies the `services` settings.
- The local entry cannot have `services`.
- `pxt://acme:main/my_app/docs` is the table `my_app/docs` in that database; `pxt.get_table()` accepts this path.
- `pxt schema` and `pxt service` take `pxt://acme:main/my_app` as the catalog directory `TARGET`.

## Deploying

To deploy, run `pxt db update`, then `pxt schema update`, then `pxt service update`.

- `pxt db update` uploads the project files and applies the entry's settings and capacity.
- `pxt db update` rebuilds the image when `uv.lock`, `requirements.txt`, `pyproject.toml`, the Python version, the
  system packages or the Pixeltable version has changed.
- `pxt db update` creates no tables and starts no services.
- The first `pxt db update` for a URI creates the database.
- The upload contains `uv.lock`, `requirements.txt`, `pyproject.toml`, and the project files not excluded by
  `.gitignore`.
- The entry's `include` and `exclude` keys adjust the selection by `.gitignore`.
- Without `uv.lock` or `requirements.txt`, `pxt db update` warns that the image contains only Pixeltable.
- Lowering `cpu`, `memory_mb`, `disk_gb` or `workers` is a destructive change and needs `--allow-destructive`.
- `pxt db update` asks for confirmation, then waits until the change is done, showing its progress on stderr.
- Ctrl-C (exit 130) or SIGTERM (exit 143) stops only the wait; Pixeltable Cloud still finishes the change.
- An agent's shell tool sends SIGTERM when a command times out.
- After the interruption, the command prints `next: pxt db status pxt://acme:main`, and rerunning `pxt db update`
  reports the accepted change.
- After three hours, `pxt db update` stops waiting and exits 2.
- `--no-wait` returns once Pixeltable Cloud accepts the change.
- When the update fails, `pxt db update` exits 1 with `next: pxt db retry pxt://acme:main`.
- `pxt service retry` retries a failed service.
- A change to the entry's settings alone, such as `vars` or a provider section, is applied without uploading the
  project.
- On `PAYMENT_REQUIRED`, `PLAN_LIMIT` or `STORE_FULL`, the command prints Pixeltable Cloud's reason verbatim; a retry
  fails until the organization's plan, payment or storage changes.
- `pxt schema update app.py pxt://acme:main/my_app` exits 1 and changes nothing while a project file with a UDF called
  by the models differs from its uploaded copy; the `next:` line gives the `pxt db update` command.
- `pxt service update app.py pxt://acme:main/my_app` refuses to start a service before the first `pxt db update`, and
  while the project differs from the version applied by the last `pxt db update`.
- `pxt service update` waits until each started service answers requests.
- `pxt service restart pxt://acme:main/my_app/ingest` restarts a hosted service on its current code;
  `pxt service update` moves it to the code of the last `pxt db update`.
- `pxt service run` refuses a `pxt://` target: it serves from the local process only.

## Secrets

- A provider key is a secret. `pxt secret set pxt://acme OPENAI_API_KEY=...` sets it for every database of the
  organization.
- `pxt secret set pxt://acme:main OPENAI_API_KEY=...` sets it for one database; the database's secret takes precedence
  over the organization's.
- A secret never goes in the project configuration.
- For a hosted database, an `api_key` or another credential in any section of the project configuration makes
  `pxt db update` and `pxt service update` fail; the error gives the `pxt secret set` command to run instead.
- The database and each hosted service read their secrets when they start.
- After `pxt secret set` or `pxt secret delete`, `pxt db restart` restarts the database, and `pxt service restart`
  restarts a service.
- `pxt secret set` prints a `next:` line for each running database and service.
- `pxt secret list` prints the names of the secrets, never their values.

## Media

- A hosted database stores inserted and computed media in its home bucket, `pxt://acme:main/buckets/home`, unless a
  column sets `destination=` or the database entry sets `db_input_media_dest` or `db_output_media_dest`.
- A media cell's `fileurl` is `pxt://acme:main/buckets/home/<key>`; the key is the part after `buckets/home/`.
- `https://storage.pixeltable.com/acme/main/buckets/home/<key>` opens the file after the user signs in to Pixeltable.
- In that URL, each segment of the key is URL-encoded.
- A script sends the API key to that URL as `Authorization: Bearer` and follows the redirect.
- That URL refuses `X-api-key`, because HTTP clients forward the header to the redirect target.
- `POST https://storage.pixeltable.com/acme/main/sign` signs keys; send the API key as `X-api-key` or
  `Authorization: Bearer`, and the body `{"keys": [...], "expires_in": 3600}`.
- The sign response is `{"urls": {key: url}, "expires_at": ...}`.
- One sign request takes 1 to 100 keys; `expires_in` ranges from 1 second to 7 days and defaults to 3600 seconds.
- Anyone with a signed URL can read the file until it expires.
- A service route returns a media output as a URL signed for one hour (`pxt guide serving`).

## Calling services and reading logs

- `pxt service list pxt://acme:main` prints the URL of each service in the database.
- Every route of a hosted service except `/health` requires a Pixeltable API key in the `X-api-key` header; the
  service's OpenAPI schema declares this.
- `pxt service logs pxt://acme:main/my_app/ingest` prints the service's log records, including requests, merged with
  its console output.
- The console output of a hosted service contains the traceback of a failed start.
- A log line appears a few seconds after it is written.
- `pxt db logs pxt://acme:main` reads the database's log.
- `pxt service logs` and `pxt db logs` read the last hour by default.

```bash
pxt login
pxt whoami
pxt db diff pxt://acme:main --add-config-entry
pxt db update pxt://acme:main -f
pxt schema update app.py pxt://acme:main/my_app
pxt service update app.py pxt://acme:main/my_app -f
pxt service list pxt://acme:main
pxt secret set pxt://acme:main OPENAI_API_KEY=sk-example
pxt db restart pxt://acme:main
pxt service restart pxt://acme:main/my_app/ingest
pxt service logs pxt://acme:main/my_app/ingest --since 10m --tail 50
pxt db logs pxt://acme:main
pxt db status pxt://acme:main
```
