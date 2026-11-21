# UDF servers

Pixeltable Cloud deploys a server for each UDF in the `pixeltable.udf` setting, with the resources of the UDF's entry,
such as a GPU. Pixeltable sends each call of the UDF to its server. The servers run on Modal; section 3 describes the
changes for other providers, such as Baseten.

For example, this entry runs `app.vision.detect` in up to four containers, each with a T4 GPU:

```toml
# pixeltable.toml
[pixeltable.udf.'app.vision.detect']
gpu = 'T4'
max_containers = 4
timeout_seconds = 300
```

The implementation is in `pixeltable/serving/udf_server.py`, `pixeltable/service/udf_servers.py` and
`pixeltable/exec/expr_eval/modal_client.py`, and in pixeltable-cloud's `pixeltable_cloud/orchestrator/modal_deploy.py`
and `pixeltable_cloud/orchestrator/actions/database/udf_deploy.py`.

## 1. Goals

- **No code changes.** A table definition, query or insert that calls the UDF is the same with and without the
  setting. A call behaves like a local one: the UDF sees the caller's default time zone, the media files go with the
  request, and the UDF's exception becomes an error in the call's rows.
- **Available for local and hosted databases.** `pxt db update pxt://org:db` deploys a hosted database's servers, and
  `pxt db update` deploys the local database's servers. A local database's tables stay on the machine; only the UDF
  calls leave it.
- **Low call overhead.** A call takes one HTTP round trip.
- **Provider-neutral core.** The call protocol, the server and the endpoint lookup don't depend on Modal.

## 2. Components

The two tables below list the steps of a deploy and of a call. Each step runs in one of three places: a `pixeltable`
process, the control plane or Modal.

### Deploy

`pxt db update`, for a hosted database or for the local one:

| # | `pixeltable` | Control plane | Modal |
| --- | --- | --- | --- |
| 1 | `pxt db update` packages the project and sends `PrepareUpdate` | | |
| 2 | | returns an upload URL, unless it has the archive | |
| 3 | uploads the archive and sends `SubmitUpdate` | | |
| 4 | | accepts the update; registers a new local database with the lifecycle `LOCAL` | |
| 5 | | builds the release image and pushes it to ECR | |
| 6 | | `DEPLOY_UDF_SERVERS` mints the release's API key and calls `modal_deploy.deploy_release()` | |
| 7 | | | deploys the App, with one Server per UDF, and returns each Server's URL |
| 8 | | records the URLs in `DbReleaseRow.udf_servers` | |
| 9 | | hosted database: writes the URLs and headers into the Kubernetes Secret `pxt-udf-<release_key[:12]>` and rolls out the release's pods with it | |
| 10 | | marks the update's receipt observed | |
| 11 | `pxt db update` waits for the observed receipt; for the local database, it drops the process's endpoints | | |

### Call

Steps 2 to 4 run only if `Env` has no endpoints, and steps 7 to 10 only if the Server has no running container.

| # | `pixeltable` | Control plane | Modal |
| --- | --- | --- | --- |
| 1 | expression evaluation sends a call of the UDF to its `ModalScheduler`, whose `ModalServerClient` needs the endpoints | | |
| 2 | a hosted pod reads the endpoints from its environment and continues at step 5; a local process sends `GetUdfServers` | | |
| 3 | | returns the current release's endpoints and fingerprint, and records a use of the database | |
| 4 | the local process compares its project's fingerprint with the release's | | |
| 5 | the endpoints go into `Env`, and `ModalServerClient` POSTs the call to the UDF's server URL | | |
| 6 | | | the proxy checks the proxy token and forwards the POST to a running container; without one, it answers 503 and starts a container |
| 7 | | | the new container's `ModalServer` sends `GetArchive`, with the release's API key |
| 8 | | returns the archive's download URL and the project's config | |
| 9 | | | `ModalServer` downloads and unpacks the archive, writes the project's config and starts `UdfServer` |
| 10 | a `ModalServerClient` GET reaches the container, and the client sends the POST again | | |
| 11 | | | `UdfServer` checks `Pxt-Udf-Secret`, runs the UDF and returns the result |
| 12 | the client puts the values into the call's rows, or turns the exception into an error in them; a status other than 200 drops the endpoints from `Env` | | |

### Server

`UdfServer` serves one UDF's calls over HTTP, with hypercorn. It runs one call at a time and picks the UDF's overload
by the request's signature. The response has the values or the UDF's exception, the call's duration and the
container's ID. `UdfServer` answers 401 to a POST without the release's secret in the `Pxt-Udf-Secret` header, and
404 to every GET.

`ModalServer`, the entrypoint of a Modal container, runs `UdfServer` on port 8000. The container's environment
variables specify the UDF, the database, the release's secret, the archive pin, the API key for `GetArchive` and the
user secrets.

### Call protocol

`UdfCallRequest` and `UdfCallResponse` define the request and response bodies, in the encoding of `proxy_protocol`:
a JSON head, followed by binary parts for media files. A request has one row's arguments or, for a batched UDF, one
value per row for each argument.

### Client

Expression evaluation gives each UDF in the setting its own resource pool, run by a `ModalScheduler`. The scheduler
runs at most `max_containers` calls at a time.

`udf_servers.udf_endpoints()` returns the process's `UdfEndpoints`: each UDF's server URL, and the headers for every
request. In a local process, the lookup costs a project fingerprint (about 100 ms) and a control plane request, so
`Env` keeps the result. Deploy step 11, call step 12 and a re-import of the project's code drop it.

While calls are active, the client sends a GET at least every second. After a 503, it cancels the POSTs that can't
complete. The cancellation works around an httpcore defect, which leaves such a POST waiting until its read timeout.

### Control plane

- **Release.** The planner deploys a release's UDF servers before any pod starts on the release:
  `_udf_servers_step()` comes before the infrastructure and rollout steps. A release with a `pixeltable.udf` setting
  is runnable once `DbReleaseRow.udf_servers` has the URLs.
- **App.** The release's Modal App is `pxt-<db_id>-<release_key[:12]>`, so a repeated deploy replaces the App of an
  interrupted one.
- **Servers, not functions.** A call to a Modal Server takes one round trip through Modal's proxy: 74 ms at p50 for
  an echo, against 153-165 ms for a Modal function's `.remote()`. Callers need only a proxy token, which can't deploy
  Apps.
- **Provider account.** `ModalAccount` has the environment's Modal credentials, from Secrets Manager: the API token
  for deploys, the proxy token for callers, the Modal secret for pulls from ECR, and the key for the release secrets.
- **Authorization.** The proxy token is the same for every App in the workspace. The release's secret, an HMAC of
  the database ID and the release key, limits a caller to one release.
- **Local databases.** The installation's local database is `local-main-<hash>` in the control plane and has no
  pods. The control plane deletes a local database and its servers after 30 days without an update or a
  `GetUdfServers` request.
- **Lifetime.** After a change to the user secrets, the next update deploys the servers again. Release collection
  stops an unreferenced release's servers and revokes its API keys. Deleting a database stops all of its servers.
