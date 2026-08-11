# Dashboard Design

Product and visual source of truth for the Pixeltable local dashboard. For stack, API, and file layout, see [ARCHITECTURE.md](ARCHITECTURE.md).

**For AI agents:** Before changing anything under `dashboard/` (or dashboard-facing UI APIs), read this file and follow it. Preserve the established theme and UX; prefer consistency over novelty. If you propose a visual or interaction departure, call it out explicitly and justify it against this document — do not silently diverge.

This document locks decisions relative to the pre–[#1494](https://github.com/pixeltable/pixeltable/pull/1494) local UI (colored, scannable sidebar) and the grayscale / KindBadge pass that shipped in the 0.7.1 wheel. **Target for `dashboard/src` going forward is this DESIGN.md**, not the wheel’s visual rebrand.

## Purpose and non-goals

**Purpose**

- Read-only local inspector for Pixeltable catalogs.
- Browse the in-process (local) catalog and optionally attach hosted catalogs (`pxt://org:db`).
- Inspect schema, data, media, lineage, history, and search.

**Non-goals**

- No writes, no auth, no admin console for Pixeltable Cloud.
- End users do not need Node.js (SPA is bundled into the Python wheel).
- Cloud catalogs are browsable attachments, not a full cloud product surface.

## Design stance

**Baseline:** the existing local dashboard visual language — colored Lucide kind icons, yellow folders, media type color (quiet scalars), green connection status, yellow computed accents.

**PR #1494** is accepted as a **feature + accessibility + bugfix** delivery, not as a full visual rebrand.

**Catalog chrome:** sticky Local / Cloud switcher above the tree (see `CatalogSwitcher`), not a stacked multi-root forest with `+ Add catalog` buried in the scroll region.

### Locked decisions

| Topic | Decision |
|-------|----------|
| KindBadge (T/V/S/R) | **Drop.** Do not combine with Lucide icons. Colored Lucide table / view / snapshot / replica icons **substitute** KindBadge as the primary kind language. |
| Sidebar `vN` | **Keep when version exists** (`version !== null`). Show `v{n}` on the right; omit when null. |
| Catalog add / switch | Sticky switcher; one active catalog’s tree in the scroll area. |
| Computed / iterator markers | Yellow Zap or colored `SquareFunction` in **schema** (and lineage) where kind icons do not apply — **not** KindBadge letters. Do **not** repeat these icons on data-grid column headers. |
| Main-content type color | **Media only** (Image / Video / Audio / Document). Light `-700`/`-800` + dark `-400` for WCAG AA on tinted pills; scalars are quiet bordered pills. Collapsed COLUMNS chips reuse the same media cue. |
| App utilities | Docs / Feedback / theme live in the **sidebar footer** above Collapse — no main-panel top utility bar. Hairline (`border-t`) separates footer from the catalog tree. |
| Object header kind | Lucide + quiet kind word (`Table` / `View` / `Snapshot` / `Replica`) — not KindBadge letters. Same in lineage DetailPanel. |
| Object header meta | Kind + name + path + **row count** only. Column breakdown stays under COLUMNS; schema/data freshness lives in **History**. |
| COLUMNS collapse | Chevron toggles expanded schema table ↔ chip summary; panel height fits that content (never a clipped 5% strip). Table switch preserves expand/collapse and always content-fits (ignore stale drag/`autoSaveId`). |
| Computed With modal | Pretty-printed expression + Raw toggle; Copy keeps the original string. Schema cell stays a compact one-liner. |
| Inspector content surface | Table detail, directory, and main content pane use one `bg-card` surface; object title is not a separate grey chrome band (light and dark). Sidebar stays tinted chrome. |
| Lineage detail CTA | Column Data Flow is a quiet bordered secondary control — **no** yellow/tan promo wash. Yellow stays for Zap / UDF emphasis only. |

## Visual principles

1. **Color carries kind** — table, view, snapshot, replica, and folder are distinguishable without reading labels.
2. **Yellow = Pixeltable accent** — folders, primary CTAs, computed emphasis. Do not strip to gray “for cleanliness.”
3. **Status is green when healthy** — connection dot stays emerald (or equivalent), not muted gray.
4. **Density over decoration** — show `vN` when a version exists; keep error marks; avoid chip piles and dual kind cues (no icon + KindBadge).
5. **Contrast without desaturation** — keep the WCAG AA `--muted-foreground` bump from #1494; do not use a11y as a reason to remove semantic color.
6. **One active catalog context** — switcher is sticky chrome; the scroll region is only that catalog’s directory tree.
7. **Main pane stays quiet** — sidebar kinds stay colored; type color in schema/data is media-only; data-grid headers are name (+ sort), not a computed/type icon strip.
8. **Kind is readable in the inspector** — object headers and directory Type columns use Lucide + kind word; directory lists say Name / Objects, not “Table” for mixed kinds.

## Information architecture

```
Header: brand left | Search + Lineage icons right    ← stacked icons when collapsed
──────────────── hairline ──────────────────────────  ← same as footer (border-t)
Catalog switcher (Local | cloud URIs | Add cloud…)   ← sticky
Filter… + Collapse all (when tree ≥ 10 nodes)        ← sticky under switcher
Directory tree (active catalog only)                 ← scroll
──────────────── hairline ──────────────────────────
Docs / Feedback / Theme                              ← sticky footer
Collapse                                             ← sticky footer
```

Main content has no top utility bar; table/schema/data share one `card` surface.

## Kind language (sidebar)

| Kind | Treatment |
|------|-----------|
| Directory | Yellow folder / open-folder Lucide |
| Table | Blue table icon (left) |
| View | Purple eye icon (left) |
| Snapshot | Orange camera icon (left) |
| Replica | Muted copy icon (left) |
| Version | `v{n}` flush right when `version !== null` |
| Errors | Warning mark when `error_count > 0` |

Do **not** use square-letter KindBadge glyphs in the tree.

**Object header / directory list:** same Lucide colors plus a quiet kind word. Object header meta is path + row count only (no column count or last-change timestamps — those live in COLUMNS / History). Directory column is **Name** (not “Table”); summary counts are **Objects** with a tables/views breakdown.

## PR #1494 accept / reject / adapt

### Accept (fits this design)

- Multi-catalog browse, `pxt-catalogs` / `pxt-active-catalog` persistence, fetch of the **active** catalog tree
- Cross-catalog search and `unavailable` reporting
- Path-safe routing (`tableHref` / `dirHref`) and path-fetched directory views
- `is_iterator_col`, richer schema Info, `Computed With`, mutable / computed / stored counts
- Table lineage + Column lineage split and deeper walk
- Pagination default 25
- Backend iterator / `GeneratingFunction.name` fixes
- `darkMode: 'class'` and muted-foreground contrast (+ `--muted-foreground-legacy`)
- Sticky catalog switcher placement (adaptation of #1494 catalog UX)

### Reject (does not fit)

- KindBadge T/V/S/R (drop entirely; Lucide substitutes; no combine)
- Gray folders / grayscale kind language
- Removal of colored **kind** accents (sidebar Lucide) or media type color
- Desaturated connection status
- Removing sidebar `vN` when version exists
- `+ Add catalog` inside the scrolling tree / stacked multi-root forest as default chrome
- Rainbow scalar type badges / computed icons on every data-grid header
- Main-panel top bar for Docs / Feedback / theme (those belong in the sidebar footer)

### Adapt

- Catalog UX → sticky switcher, one active tree (`CatalogSwitcher` + active-catalog fetch)
- Schema header → keep #1494’s cleaner Info model; type badges colored for **media** only; scalars quiet
- Schema columns → compact Name/Type defaults; **Info** = sparse secondary metadata (comment / validation / unstored / destination), **hidden when unused** so Computed With gets the width
- Computed / iterator → yellow Zap or colored `SquareFunction` in schema/lineage, not KindBadge letters; omit from data-grid headers
- Inspector chrome → one `card` content surface (header + schema + data); kind word beside Lucide in headers; honest directory object counts
- Lineage DetailPanel → quiet Column Data Flow CTA; Zap/UDF yellow kept; no yellow promo wash (esp. light mode)
- Search unavailable → collapsed one-line “skipped” summary by default; path + short reason on expand; raw exception in `title`

## Component ownership (intended look)

| Concern | Source of truth |
|---------|-----------------|
| Tree icons + `vN` | [`src/components/DirectoryTree.tsx`](src/components/DirectoryTree.tsx) — `DirectoryTreePanel`: sticky Filter/Collapse; colored `getNodeIcon`; `vN` when version exists; no KindBadge |
| Catalog switcher | [`src/components/CatalogSwitcher.tsx`](src/components/CatalogSwitcher.tsx) |
| Catalog persistence | [`src/lib/utils.ts`](src/lib/utils.ts) (`pxt-catalogs` / active catalog) |
| Column type colors | [`src/lib/column-types.tsx`](src/lib/column-types.tsx) — media AA light/dark; quiet scalars; collapsed chip parity |
| Schema column widths / Info | [`src/components/TableDetailView.tsx`](src/components/TableDetailView.tsx) (`ColumnChips`) — compact Name/Type; Info hidden when empty |
| Func accents | [`src/lib/func-styles.ts`](src/lib/func-styles.ts) |
| Shell / routing / footer utils | [`src/App.tsx`](src/App.tsx) |
| Search unavailable | [`src/components/SearchPanel.tsx`](src/components/SearchPanel.tsx) — collapsed by default |

The PyPI 0.7.1 bundled SPA may still show KindBadge + in-tree Add catalog. Treat that as divergent until the wheel is rebuilt from sources that follow this doc.

## Future work

Kind language, sticky switcher, accents, and inspector `card` surface match this doc. Remaining polish only:

1. Broader light/dark surface ladder (Search / Lineage / Welcome vs chrome).
2. Dense data-toolbar wrap at mid widths.
