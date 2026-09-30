---
applyTo: "pixeltable/serving/**,pixeltable/service/**,pixeltable_cli/**,pixeltable/runtime.py,pixeltable/utils/cloud_utils.py,pixeltable/utils/app_module.py,pixeltable/utils/project.py"
---

# Serving, CLI and hosted review

- Support local and hosted catalogs. Hosted column metadata uses `col.col_md`, not local `col.col`.
  Hosted media responses need accessible HTTP URLs, not private filesystem paths or storage URIs.
  Generated service URLs must respect gateway `root_path`.
- Preserve API-key precedence and resolve fresh credentials on reconnect. Bind organization lookup and
  renewal to the same account/session; check that identity under the cache lock. Before caching or sending
  an organization-targeted token, verify its `org_id` matches the requested organization. Persist rotated
  refresh tokens for the same session even when rejecting the access token's scope. Do not report successful
  login after removing its session. Retry only operations safe to repeat; distinguish transient failures
  from rejected credentials.
- Project loading/packaging must exclude Python environments, including an in-project `.venv`;
  `.gitignore` alone is insufficient. Keep application files under the project root and preserve installed
  dependencies during reload.
- Validate external input with typed errors. Production responses use `e.message` and omit diagnostic
  `detail`; keep secrets and private paths out of responses. Internal invariant assertions remain valid.
- Close owned resources on success and failure; async clients close on their owning loop. Preserve adopted
  loops, bound shutdown waits and avoid joining the current worker.
  FastAPI/ASGI liveness handlers should be asynchronous and avoid blocking work; this does not apply to
  synchronous daemon handlers.
