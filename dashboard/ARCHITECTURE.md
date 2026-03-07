# Dashboard Architecture

Read-only local UI for inspecting Pixeltable databases. No writes, no auth.

## Stack

**Backend:** stdlib `ThreadingHTTPServer` — one thread per request, no async, no deps. Binds `127.0.0.1:8080`.
**Frontend:** React 18 + Vite + TypeScript + Tailwind. `@xyflow/react` + `dagre` for DAGs. No state library.

## Backend (`pixeltable/dashboard/`)

| File | Role |
|------|------|
| `__init__.py` | `pxt.dashboard.serve()` — opens browser + starts server |
| `server.py` | HTTP routing, static serving, CORS, error suppression |
| `bridge.py` | Pixeltable → JSON. Shared: `_build_select`, `_format_versions`, `_resolve_fileurl` |

## Frontend (`dashboard/src/`)

| File | Role |
|------|------|
| `App.tsx` | Sidebar, routing (`/`, `/table/*`, `/dir/*`, `/lineage`), theme toggle |
| `api/client.ts` | Typed fetch wrappers |
| `types/index.ts` | TS interfaces matching API shapes |
| `hooks/useDebounce.ts` | Debounce hook |
| `lib/column-types.tsx` | Type→icon/color map |
| `lib/python-highlight.tsx` | Python syntax highlighter |
| `lib/column-lineage.ts` | Column dependency DAG builder |
| `components/TableDetailView.tsx` | Data/Lineage/History tabs, lightbox, JSON viewer, filters, pagination, export |
| `components/PipelineInspector.tsx` | Pipeline graph + node finder + detail sidebar |
| `components/ColumnFlowDiagram.tsx` | Per-table column DAG |
| `components/DirectoryTree.tsx` | Sidebar explorer with error indicators |
| `components/SearchPanel.tsx` | Cmd+K search |

## API (GET only)

| Endpoint | Returns | Params |
|----------|---------|--------|
| `/api/health` | `{status, version}` | — |
| `/api/dirs` | Directory tree + error counts | — |
| `/api/status` | Version, config, total_tables, total_errors | — |
| `/api/search` | Matching dirs, tables, columns | `q`, `limit` (50, max 100) |
| `/api/pipeline` | DAG nodes + edges | — |
| `/api/tables/{path}` | Schema, columns, indices, versions | — |
| `/api/tables/{path}/data` | Paginated rows, media URLs, per-cell errors | `offset`, `limit` (50, max 500), `order_by`, `order_desc`, `errors_only` |
| `/api/tables/{path}/export` | CSV download | `limit` (100k default, 1M max) |

## User Flows

1. **Navigate** — sidebar tree → directory summary or table detail
2. **Search** — Cmd+K spotlight → keyboard navigate → Enter
3. **Schema** — collapsible column chips or expanded table with expressions
4. **Data** — server-side sort, SQL OFFSET pagination, client-side filters (current page)
5. **Media** — thumbnails → lightbox with arrow nav; unstored PIL → base64
6. **JSON** — truncated cells → tree viewer with search + path copy
7. **Lineage** — per-table column DAG + full pipeline graph with node finder
8. **History** — per-table version tab (inserts/updates/deletes/errors)
9. **Export** — CSV (100k default), SDK snippet copy
10. **Live** — auto-refresh (10s), manual refresh

## Key Decisions

**Sort:** server-side `query.order_by()` → SQL. **Filter:** client-side on current page only. **Pagination:** SQL OFFSET (`query.limit(n, offset=k)`); deep pages slow. `errors_only` returns page-size total. **Media:** stored `file://` → HTTP proxy, unstored PIL → base64, external → passthrough. **Errors:** `BrokenPipeError`/`ConnectionResetError` silenced; `PixeltableWarning` suppressed during API calls. **CSV:** media → URLs, JSON → strings.

## Auto-start

`pxt.init()` spawns server in a daemon thread. On by default.

| Control | Effect |
|---------|--------|
| `dashboard=False` | disable this session |
| `dashboard=True` | force-start |
| `dashboard_port=9090` | custom port |
| `PIXELTABLE_DASHBOARD=0` | disable via env |
| `PIXELTABLE_DASHBOARD_PORT=N` | env port override |

Port conflicts auto-detected. Pre-built static ships in the wheel; no Node.js needed.

## Dev

```
cd dashboard && npm run dev   # :5173 hot reload → :8080 backend
npm run build                 # → pixeltable/dashboard/static/
```
