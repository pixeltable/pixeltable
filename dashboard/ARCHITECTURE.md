# Dashboard Architecture

Read-only UI for inspecting Pixeltable databases. The default build uses the local daemon. An optional MCP App build uses the host connection; neither build adds writes.

**Visual / UX:** [DESIGN.md](DESIGN.md). This file is stack, APIs, and runtime only.

## Stack

**Backend:** stdlib `http.server.ThreadingHTTPServer` on the pxt daemon (`pixeltable_cli/server/`). Same `127.0.0.1:22089` port the CLI uses (`PXT_PORT` overrides). Stdlib-only so the daemon ships in the base wheel without an extras install.

**Frontend:** React 18 + Vite + TypeScript + Tailwind. `@xyflow/react` + `dagre` for DAGs. `react-router-dom` for `/`, `/table/*`, `/dir/*`, `/lineage`. `react-resizable-panels` for the sidebar. No global state library.

## Backend (`pixeltable_cli/server/`)

| File | Role |
|------|------|
| `http_server.py` | ThreadingHTTPServer: `/api/*`, CORS for Vite `:5173`, static SPA fallback |
| `router.py` | Regex router with `{name:path}` converters; pydantic via `Request.body()` |
| `routes.py` | All `/api/*` endpoints, including `/api/dashboard/*` |
| `bridge.py` | Pixeltable → JSON (select/media, pipeline, data, CSV, search, status) |
| `static/` | Built SPA (`npm run build` output) |

## Frontend (`dashboard/src/`)

| Path | Role |
|------|------|
| `App.tsx` | Shell: sticky catalog switcher, sidebar tree, routes, theme, status |
| `api/client.ts` | Client factory with a per-instance transport; named HTTP exports remain compatible |
| `lib/dashboard-runtime.tsx` | Per-render client, preferences, and local or embedded capabilities |
| `embedded/` | MCP Apps bridge, fixed catalog, validated media references, MemoryRouter |
| `types/index.ts` | TS interfaces matching API shapes |
| `lib/utils.ts` | `cn()`, path hrefs, catalog localStorage helpers |
| `lib/column-types.tsx` | Column type → icon/color |
| `lib/func-styles.ts` | UDF / query / iterator accents |
| `components/CatalogSwitcher.tsx` | Local / Cloud switch + add hosted catalog |
| `components/DirectoryTree.tsx` | Active-catalog explorer |
| `components/SearchPanel.tsx` | Cmd+K search |
| `components/TableDetailView.tsx` | Schema / Data / Lineage / History |
| `components/PipelineInspector.tsx` | Active-catalog pipeline graph |
| `components/ColumnFlowDiagram.tsx` | Per-table column DAG |

## API

| Endpoint | Returns | Params |
|----------|---------|--------|
| `GET /api/status` | Version, home, db/media/cache, totals | Optional `sizes` |
| `GET /api/dirs` | Listing / tree (`LsResponse`) | `tree`, `details`, `counts`; optional `path` (`local` / omit / `pxt://…`) |
| `GET /api/dashboard/search` | Matching dirs, tables, columns | `q`; repeated `catalogs` |
| `GET /api/dashboard/pipeline` | Full DAG for the **active** catalog | Optional `path` (`local` / omit / `pxt://org:db`) |
| `GET /api/dashboard/tables/pipeline` | DAG slice for one table (local or hosted path) | `path` |
| `GET /api/dashboard/tables/meta` | Schema, indices, versions, … | `path` |
| `GET /api/dashboard/tables/data` | Paginated rows, media URLs, errors | `path`, `offset`, `limit`, `order_by`, `order_desc`, `errors_only` |
| `GET /api/dashboard/tables/export` | CSV download | `path`, `limit` (default/max 100k) |

## User Flows

1. **Switch catalog** — sticky Local / Cloud switcher; one active tree (`pxt-catalogs` / `pxt-active-catalog` in localStorage)
2. **Navigate** — sidebar tree → directory summary or table detail
3. **Search** — Cmd+K spotlight
4. **Schema** — column chips or expanded table with expressions
5. **Data** — server-side sort, SQL OFFSET pagination, client-side filters (current page)
6. **Media** — thumbnails → lightbox; `fileurl` rewritten for local files
7. **JSON** — truncated cells → tree viewer
8. **Lineage** — per-table column DAG + full pipeline graph for the **active** catalog
9. **History** — per-table version tab
10. **Export / Live** — CSV + SDK snippet; auto-refresh (10s)

## Key Decisions

**Sort:** server-side `query.order_by()` to SQL. **Filter:** client-side on current page only. **Pagination:** SQL OFFSET; deep pages slow. **Media:** select `fileurl` (not raw bytes); local `file://` → daemon HTTP. **Catalog chrome:** sticky switcher + one active tree (see DESIGN.md).

## Opening

```
pxt dashboard   # ensure daemon is running, print URL, open browser
```

## Dev & Release

```
cd dashboard && npm run dev   # :5173 hot reload, proxies /api → :22089
npm run build                 # → pixeltable_cli/server/static/
```

The normal build is bundled into the Python wheel by Hatch (`artifacts` in `pyproject.toml`). `make install` builds it; the private release script is `admin/scripts/release.sh`. End users do not need Node.js.

## Optional MCP App build

Run `npm ci`, `npm test`, and `npm run build:embedded` from `dashboard/`. This produces `pixeltable_cli/server/embedded/embedded.html` and `manifest.json`, separate from the normal SPA. The wheel does not carry this build. Pixeltable Cloud builds it from the core commit pinned in its lockfile and checks the HTML in as an MCP resource. The manifest records contract version 1, the source commit, whether dashboard sources were dirty, byte length and SHA-256 of the HTML. The build rejects HTML over 1 MiB (our budget, not a claimed host limit). A dirty dashboard tree is recorded in the manifest and refused by that release check. Local candidate builds can be inspected before a release; they are not production release artifacts.

The embedded entry uses the standard `@modelcontextprotocol/ext-apps` bridge. The host's `open_collection_explorer` or browse result provides `connection: { organization, db }` (or `page.connection`). With `db: null`, `databases: [{ db }]` supplies the database chooser. No account token is exposed to JavaScript. At widths below 640 pixels the wrapper asks the user to expand the panel or requests fullscreen through the host; it does not squeeze the desktop inspector into a clipped sidebar. Theme comes from host context. The connection fixes the catalog; catalog switching, the local status/config request, CSV export, browser persistence, auto refresh, the page-jump prompt and external links are absent. Layout and column preferences remain in memory. The normal entry retains BrowserRouter, HTTP, browser preferences and all existing controls.

`createDashboardClient(transport)` owns request construction and error handling. The MCP transport maps six exact GET routes to the app-only `read_dashboard({ db, endpoint, parameters })` tool, refuses other routes and cross-catalog requests, and surfaces host refusals as the same thrown errors. The server and gateway independently validate the same route and query allowlists; UI restrictions are not authorization. Page sizes are at most 25 in this build. Existing filters still apply only to the current page; catalog search searches names and schema, not stored cell values.

Media cells carry opaque `pxt-preview:` locators instead of public download URLs. The embedded resolver validates the scoped record reference and calls app-only `preview_collection_media`; only bounded JPEG/audio/video data URIs reach media elements. The server re-verifies table identity, version, sorting and row fingerprint before reading a stored cell. Unsupported documents and large video/audio show an explicit fallback message. Normal daemon media rendering remains unchanged. The single HTML file contains its JavaScript, CSS and logo and requires no external asset/network domains.

Tests cover client isolation, the route boundary, media validation, storage denial and host theme updates. They do not prove real ChatGPT/Claude OAuth, CSP or marketplace acceptance; both real hosts must pass those checks before enabling the endpoint. Evidence attachment, cell search coverage and agent trace presentation are later work in this same UI, not included in this foundation.
