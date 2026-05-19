# Dashboard Architecture

Read-only local UI for inspecting Pixeltable databases. No writes, no auth.

## Stack

**Backend:** FastAPI on the pxt daemon (`pxt_cli/server/`). Same `127.0.0.1:22089` port the CLI uses.
**Frontend:** React 18 + Vite + TypeScript + Tailwind. `@xyflow/react` + `dagre` for DAGs. No state library.

## Backend

| File | Role |
|------|------|
| `pxt_cli/server/app.py` | FastAPI app factory: CORS, dashboard feature gate, StaticFiles SPA mount |
| `pxt_cli/server/routes.py` | All `/api/*` endpoints, including `/api/dashboard/*` |
| `pxt_cli/server/state.py` | In-process `dashboard_enabled` flag |
| `pixeltable/dashboard/bridge.py` | Pixeltable to JSON. `_build_select`, `_resolve_fileurl`, `get_pipeline`, `get_table_data`, `export_table_csv`, `search`, `get_status` |

## Frontend (`dashboard/src/`)

| File | Role |
|------|------|
| `App.tsx` | Sidebar, routing (`/`, `/table/*`, `/dir/*`, `/lineage`), theme toggle |
| `api/client.ts` | Typed fetch wrappers |
| `types/index.ts` | TS interfaces matching API shapes |
| `hooks/useDebounce.ts` | Debounce hook |
| `lib/column-types.tsx` | Type to icon/color map |
| `lib/python-highlight.tsx` | Python syntax highlighter |
| `lib/column-lineage.ts` | Column dependency DAG builder |
| `components/TableDetailView.tsx` | Data/Lineage/History tabs, lightbox, JSON viewer, filters, pagination, export |
| `components/PipelineInspector.tsx` | Pipeline graph + node finder + detail sidebar |
| `components/ColumnFlowDiagram.tsx` | Per-table column DAG |
| `components/DirectoryTree.tsx` | Sidebar explorer with error indicators |
| `components/SearchPanel.tsx` | Cmd+K search |

## API

| Endpoint | Returns | Params |
|----------|---------|--------|
| `GET /api/dirs` | Directory tree + error counts | — |
| `GET /api/status` | Version, config, total_tables, total_errors | — |
| `GET /api/dashboard/search` | Matching dirs, tables, columns | `q`, `limit` (50, max 100) |
| `GET /api/dashboard/pipeline` | DAG nodes + edges for the full catalog | — |
| `GET /api/dashboard/tables/{path}/pipeline` | DAG slice connected to one table | — |
| `GET /api/dashboard/tables/{path}/meta` | Schema, columns, indices, versions, iterator info, media validation, destinations | — |
| `GET /api/dashboard/tables/{path}/data` | Paginated rows, media URLs, per-cell errors | `offset`, `limit` (50, max 500), `order_by`, `order_desc`, `errors_only` |
| `GET /api/dashboard/tables/{path}/export` | CSV download | `limit` (100k default, 1M max) |
| `POST /api/dashboard/control` | Toggle the feature flag | `{action: "enable"|"disable"}` |

When the dashboard flag is off, every `/api/dashboard/*` route except `/control` returns 503 and the SPA at `/` returns 404. CLI routes are unaffected.

## User Flows

1. **Navigate** - sidebar tree to directory summary or table detail
2. **Search** - Cmd+K spotlight, keyboard navigate, Enter
3. **Schema** - collapsible column chips or expanded table with expressions
4. **Data** - server-side sort, SQL OFFSET pagination, client-side filters (current page)
5. **Media** - thumbnails to lightbox with arrow nav; unstored PIL to base64
6. **JSON** - truncated cells to tree viewer with search + path copy
7. **Lineage** - per-table column DAG + full pipeline graph with node finder
8. **History** - per-table version tab (inserts/updates/deletes/errors)
9. **Export** - CSV (100k default), SDK snippet copy
10. **Live** - auto-refresh (10s), manual refresh

## Key Decisions

**Sort:** server-side `query.order_by()` to SQL. **Filter:** client-side on current page only. **Pagination:** SQL OFFSET (`query.limit(n, offset=k)`); deep pages slow. `errors_only` returns page-size total. **Media:** `fileurl` is fetched instead of downloading raw media content (fixes S3 access issues). Local `file://` to HTTP proxy, external passes through. **CSV:** media to URLs, JSON to strings.

## Control

The dashboard is off by default at daemon start. Turn it on from a terminal:

```
pxt dashboard start          # POSTs enable, prints URL, opens browser
pxt dashboard start --no-open
pxt dashboard stop
pxt dashboard restart
pxt dashboard open           # print and open the URL without changing the flag
```

`pxt dashboard start` auto-spawns the daemon if it isn't already up.

## Dev & Release

```
cd dashboard && npm run dev   # :5173 hot reload, proxies /api to :22089
npm run build                 # to pxt_cli/server/static/
```

During release (`scripts/release.sh`), the dashboard is built via `npm run build` and bundled into the Python wheel via `hatchling` (`artifacts` in `pyproject.toml`). End users do not need Node.js.
