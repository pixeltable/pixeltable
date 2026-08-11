# Dashboard Architecture

Read-only local UI for inspecting Pixeltable databases. No writes, no auth.

**Visual / UX source of truth:** [DESIGN.md](DESIGN.md). Follow that document for theme, kind language, catalog chrome, and accept/reject of prior UI changes. This file covers stack, layout, APIs, and runtime only.

## Stack

**Backend:** stdlib `http.server.ThreadingHTTPServer` on the pxt daemon (`pixeltable_cli/server/`). Same `127.0.0.1:22089` port the CLI uses (`PXT_PORT` overrides). Stdlib-only so the daemon ships in the base wheel without an extras install.

**Frontend:** React 18 + Vite + TypeScript + Tailwind. `@xyflow/react` + `dagre` for DAGs. `react-router-dom` for `/`, `/table/*`, `/dir/*`, `/lineage`. `react-resizable-panels` for the sidebar. No global state library.

## Backend (`pixeltable_cli/server/`)

| File | Role |
|------|------|
| `daemon.py` | Process lifecycle, PID, working directory sessions |
| `http_server.py` | ThreadingHTTPServer host: `/api/*` dispatch, CORS for Vite `:5173`, static SPA fallback to `static/` |
| `router.py` | Regex router with `{name:path}` converters; pydantic body validation via `Request.body()` |
| `routes.py` | All `/api/*` endpoints, including `/api/dashboard/*` |
| `bridge.py` | Pixeltable → JSON: select/media URLs, pipeline, table data, CSV export, search, status |
| `static/` | Built SPA (`npm run build` output); served for non-`/api` paths |

Media: bridge selects `.fileurl` for media columns and rewrites `file://` to the daemon HTTP address so the browser can load local files; external URLs pass through.

## Frontend (`dashboard/src/`)

| Path | Role |
|------|------|
| `main.tsx` | React mount, router |
| `index.css` | Theme CSS variables (light/dark) |
| `App.tsx` | Shell: sidebar, sticky catalog switcher, routes, theme toggle, status |
| `api/client.ts` | Typed fetch wrappers (`getDirectoryTree`, meta/data/search/pipeline/status) |
| `types/index.ts` | TS interfaces matching API shapes |
| `hooks/useDebounce.ts` | Debounce for search / filters |
| `lib/catalogs.ts` | `LOCAL_CATALOG`, `pxt-catalogs` / active-catalog localStorage |
| `lib/utils.ts` | `cn()` class merge |
| `lib/column-types.tsx` | Column type → icon/color map |
| `lib/func-styles.ts` | Computed-function kind accents (UDF / query / iterator) |
| `lib/python-highlight.tsx` | Python syntax highlighter |
| `lib/column-lineage.ts` | Column dependency DAG builder |
| `components/CatalogSwitcher.tsx` | Sticky Local / Cloud catalog switcher + add/remove |
| `components/DirectoryTree.tsx` | Active-catalog explorer (filter, collapse-all, errors, `vN`) |
| `components/SearchPanel.tsx` | Cmd+K search |
| `components/TableDetailView.tsx` | Schema / Data / Lineage / History, media lightbox, export |
| `components/PipelineInspector.tsx` | Full-catalog pipeline graph |
| `components/ColumnFlowDiagram.tsx` | Per-table column DAG |

Docs at repo root of this package: [DESIGN.md](DESIGN.md), this file.

## Multi-catalog

- **Local** catalog is always available (in-process Pixeltable home).
- Extra hosted catalogs (`pxt://org:db`) are stored in `localStorage` (`pxt-catalogs`); active selection in `pxt-active-catalog`.
- Sticky **CatalogSwitcher** sits above the scrollable tree (not inside it). The tree shows **only the active catalog**.
- `getDirectoryTree(catalogPath?)` loads the active root; hosted paths are passed as `path` on `/api/dirs`.
- Search may pass saved hosted URIs as repeated `catalogs` query params when the daemon supports multi-catalog search.

## API (dashboard-relevant)

| Endpoint | Returns | Params / notes |
|----------|---------|----------------|
| `GET /api/status` | Flat status (`pxt_version`, home, db_url, media/cache dirs, totals) | Optional `sizes`; client maps to nested `{version, config}` |
| `GET /api/dirs` | Listing / tree (`LsResponse`) | `tree`, `details`, `counts`; optional `path` (`local` / omit = in-process root; `pxt://…` or in-catalog path) |
| `GET /api/dashboard/search` | Matching dirs, tables, columns (+ `unavailable` when present) | `q`; repeated `catalogs` for hosted URIs |
| `GET /api/dashboard/pipeline` | Full-catalog DAG nodes + edges | — |
| `GET /api/dashboard/tables/pipeline` | DAG slice for one table | `path` |
| `GET /api/dashboard/tables/meta` | Table metadata (schema, indices, versions, …) | `path` |
| `GET /api/dashboard/tables/data` | Paginated rows, media URLs, per-cell errors | `path`, `offset`, `limit`, `order_by`, `order_desc`, `errors_only` |
| `GET /api/dashboard/tables/export` | CSV download | `path`, `limit` (default/max 100k) |

CLI-oriented routes under `/api/tables/*`, `/api/columns`, `/api/indexes`, etc. share the same daemon but are not required by the SPA.

## User Flows

1. **Switch catalog** — sticky switcher: Local, saved cloud URIs, or Add cloud catalog (`pxt://…`)
2. **Navigate** — sidebar tree → directory summary or table detail
3. **Search** — Cmd+K spotlight, keyboard navigate, Enter
4. **Schema** — collapsible column chips or expanded table with expressions
5. **Data** — server-side sort, SQL OFFSET pagination, client-side filters (current page)
6. **Media** — thumbnails → lightbox; `fileurl` rewritten for local files
7. **JSON** — truncated cells → tree viewer with search + path copy
8. **Lineage** — per-table column DAG + full pipeline graph with node finder
9. **History** — per-table version tab (inserts/updates/deletes/errors)
10. **Export** — CSV (100k default), SDK snippet copy
11. **Live** — auto-refresh (10s), manual refresh

## Key Decisions

**Sort:** server-side `query.order_by()` to SQL. **Filter:** client-side on current page only. **Pagination:** SQL OFFSET (`query.limit(n, offset=k)`); deep pages slow. `errors_only` returns page-size total. **Media:** `fileurl` is selected instead of downloading raw media (fixes remote object access). Local `file://` → HTTP via daemon address; external URLs pass through. **CSV:** media to URLs, JSON to strings. **Catalog chrome:** sticky switcher + one active tree (see DESIGN.md).

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
