# Dashboard Architecture

Read-only local UI for inspecting Pixeltable databases. No writes, no auth.

## Stack

**Backend:** Python stdlib `http.server.ThreadingHTTPServer`. One thread per request,
no async, no third-party server. Binds `127.0.0.1:8080`.

**Frontend:** React 18 + Vite + TypeScript + Tailwind CSS. `lucide-react` icons.
`dagre` + `@xyflow/react` for DAG layout. No state library — just hooks.

## Files

### Backend (`pixeltable/dashboard/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Public API: `pxt.dashboard.serve()` opens browser + starts server |
| `server.py` | HTTP routing, static file serving, CORS, error suppression |
| `bridge.py` | Translates Pixeltable internals → JSON dicts. Shared helpers: `_build_select`, `_resolve_fileurl`, `_is_media_type` |

### Frontend (`dashboard/src/`)

| File | Purpose |
|------|---------|
| `main.tsx` | React entry point |
| `App.tsx` | Sidebar layout, routing (`/`, `/table/*`, `/lineage`), search modal |
| `api/client.ts` | Typed `fetch` wrappers for all API endpoints |
| `types/index.ts` | TypeScript interfaces matching every API response shape |
| `hooks/useApi.ts` | `useDebounce` hook |
| `lib/utils.ts` | `cn()` — Tailwind class merger |
| `lib/column-types.tsx` | Single source for type→icon/color mapping (used by TableDetail + ColumnFlowDiagram) |
| `lib/func-styles.ts` | UDF function type styling (builtin/custom/query) |
| `lib/python-highlight.tsx` | Lightweight Python syntax highlighter (used by schema + pipeline) |
| `lib/column-lineage.ts` | Builds ReactFlow DAG from column dependency metadata |
| `components/TableDetailView.tsx` | **Main workhorse.** Table/gallery views, media lightbox, JSON viewer, filters, pagination, CSV export, schema tab, expanded row modal |
| `components/DirectoryTree.tsx` | Sidebar explorer tree with error indicators |
| `components/PipelineInspector.tsx` | Full-page lineage graph + column detail sidebar |
| `components/ColumnFlowDiagram.tsx` | Per-table column DAG (used in both TableDetail and Pipeline) |
| `components/SearchPanel.tsx` | Cmd+K search across dirs, tables, columns |

## API Endpoints

All `GET`-only. Defined in `server.py`, implemented in `bridge.py`.

| Endpoint | What it returns |
|----------|-----------------|
| `/api/health` | `{ status, version }` |
| `/api/dirs` | Full directory tree with error counts |
| `/api/status` | System config: home, DB URL, cache paths, table count |
| `/api/search?q=` | Matching dirs, tables, columns |
| `/api/pipeline` | DAG nodes + edges for all tables (lineage view) |
| `/api/tables/{path}` | Schema: columns, indices, versions, base table |
| `/api/tables/{path}/data?offset=&limit=&order_by=&order_desc=` | Paginated rows with resolved media URLs |
| `/api/tables/{path}/errors?limit=` | Per-column error counts + sample messages |
| `/api/tables/{path}/export` | CSV file download (default 100k rows) |

## User Flows

1. **Navigate**: Sidebar directory tree (filterable, collapsible) → click table → schema + data
2. **Search**: ⌘K spotlight across dirs, tables, columns → keyboard navigate → Enter to open
3. **Schema**: Collapsible column chips (compact pills or expanded table with expressions, indices)
4. **Data**: Table view with server-side sort, pagination, client-side filters + text search (current page only). Gallery grid auto-appears for media-heavy tables.
5. **Media**: Cell thumbnails → click → fullscreen lightbox with ←→ arrow navigation. Unstored PIL images render as inline base64. External docs open in new tab.
6. **JSON**: Truncated JSON cells → expandable tree viewer with search, expand/collapse, and path copy
7. **Row Detail**: Gallery card → expanded row modal (media + all fields), ←→ navigation
8. **Lineage**: Per-table tab: base→current→derived chain + column dependency DAG. Full-page pipeline graph of all tables.
9. **Export**: CSV download (up to 100k rows), Python SDK snippet copy
10. **Live Monitoring**: Auto-refresh toggle (10s polling), manual refresh, timestamp

## Key Decisions

**Sorting: server-side.** `order_by` → Pixeltable `query.order_by()` → SQL. Correct across pages.

**Filtering + search: client-side.** `useMemo` on current page rows only. UI shows
"(this page)". Pushing server-side would mean translating 4 filter types into
`where()` clauses — not worth it for v1.

**Pagination: SQL OFFSET.** `query.limit(n, offset=k)`. Fast for early pages; deep
offsets are slow (SQL scans all preceding rows). Keyset pagination needs API changes.

**Media URLs: three paths.** (1) Stored: `file://` → HTTP via media server.
(2) Unstored PIL: inline base64. (3) External: passthrough (docs open in new tab).

**Error suppression.** `BrokenPipeError` silenced at server + handler level.
`PixeltableWarning` suppressed during API calls.

**CSV export.** Media → URLs, JSON → strings. Default limit 100k rows.

## Packaging & Deployment

**Users need zero frontend tooling.** The SPA is pre-built to `pixeltable/dashboard/static/`
and shipped inside the Python wheel as a build artifact (`pyproject.toml: artifacts`).
`pip install pixeltable` includes the static files — no Node.js, no npm, no build step.

The `dashboard/` source directory (TypeScript, React, Tailwind) is excluded from the
Python package. It only exists for development.

If the `static/` directory is missing (dev checkout without a build), the server renders
a fallback HTML page with build instructions instead of crashing.

## Auto-start

`pxt.init()` spawns the server in a daemon thread (`globals.py`). On by default (like Ray).

| Control | Effect |
|---------|--------|
| `pxt.init(dashboard=False)` | Disable for this session |
| `pxt.init(dashboard=True)` | Force-start (overrides env var) |
| `PIXELTABLE_DASHBOARD=0` | Disable via env var |
| `PIXELTABLE_DASHBOARD_PORT=9090` | Change default port |

Port conflicts are auto-detected: reuses existing Pixeltable dashboards, picks a free
port if occupied by another service.

## Dev

```
cd dashboard && npm run dev    # hot reload on :5173, proxied to :8080
npm run build                  # production build → pixeltable/dashboard/static/
python _start_dashboard.py     # standalone backend
```
