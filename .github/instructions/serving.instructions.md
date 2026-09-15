---
applyTo: "pixeltable/serving/**,pixeltable/service/**,pixeltable_cli/server/**"
---

This code runs on a hosted pod as well as locally. The same class of bug appeared four times in the last 100 PRs.

- Do not assume a local catalog. `col.col` resolves through it and is empty for a hosted table; use `col.col_md.name`.
- Do not assume a local filesystem. Pod storage is ephemeral; media belongs in the home bucket, and results return presigned HTTP URLs, never a raw `pxtfs://` or `s3://` URI.
- Do not assume the request path. The gateway strips the service prefix, so URLs need `root_path`.
- Any traversal rooted at `Config.get().project_root` must exclude the Python environment. `.gitignore` is not enough; an in-project `.venv` caused two regressions.
- No `assert` in a request handler. It vanishes under `-O`.
- Send `e.message` to a client, not `str(e)`, which appends the detail. `Error.detail` does not cross to a remote client.
- Release resources explicitly; never rely on GC. Sockets and async clients belong in `with`/`closing()`, and their owner needs a `close()`.
- No unbounded `join()`, `result()`, or `wait()` in a shutdown path. Give it a timeout.
- A health or liveness endpoint must be `async def`, or it starves on the threadpool under load.
