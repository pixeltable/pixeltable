# Dashboard Architecture

Read-only local UI for inspecting Pixeltable databases. No writes, no auth.

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
| `api/client.ts` | Typed fetch wrappers |
| `types/index.ts` | TS interfaces matching API shapes |
| `lib/utils.ts` | `cn()`, path hrefs, catalog localStorage helpers |
| `lib/column-types.tsx` | Column type → icon/color |
| `lib/func-styles.ts` | UDF / query / iterator accents |
| `components/CatalogSwitcher.tsx` | Local / Cloud switch + add hosted catalog |
| `components/DirectoryTree.tsx` | Active-catalog explorer |
| `components/SearchPanel.tsx` | Cmd+K search |
| `components/TableDetailView.tsx` | Schema / Data / Lineage / History |
| `components/PipelineInspector.tsx` | Full-catalog pipeline graph |
| `components/ColumnFlowDiagram.tsx` | Per-table column DAG |

## API

| Endpoint | Returns | Params |
|----------|---------|--------|
| `GET /api/status` | Version, home, db/media/cache, totals | Optional `sizes` |
| `GET /api/dirs` | Listing / tree (`LsResponse`) | `tree`, `details`, `counts`; optional `path` (`local` / omit / `pxt://…`) |
| `GET /api/dashboard/search` | Matching dirs, tables, columns | `q`; repeated `catalogs` |
| `GET /api/dashboard/pipeline` | Full-catalog DAG | — |
| `GET /api/dashboard/tables/pipeline` | DAG slice for one table | `path` |
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
8. **Lineage** — per-table column DAG + full pipeline graph
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

During release (`scripts/release.sh`), the dashboard is built via `npm run build` and bundled into the Python wheel via `hatchling` (`artifacts` in `pyproject.toml`). End users do not need Node.js.
