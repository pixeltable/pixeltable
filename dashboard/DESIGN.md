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

**Baseline:** the existing local dashboard visual language — colored Lucide kind icons, yellow folders, semantic type colors, green connection status, yellow computed accents.

**PR #1494** is accepted as a **feature + accessibility + bugfix** delivery, not as a full visual rebrand.

**Catalog chrome:** sticky Local / Cloud switcher above the tree (see `CatalogSwitcher`), not a stacked multi-root forest with `+ Add catalog` buried in the scroll region.

### Locked decisions

| Topic | Decision |
|-------|----------|
| KindBadge (T/V/S/R) | **Drop.** Do not combine with Lucide icons. Colored Lucide table / view / snapshot / replica icons **substitute** KindBadge as the primary kind language. |
| Sidebar `vN` | **Keep when version exists** (`version !== null`). Show `v{n}` on the right; omit when null. |
| Catalog add / switch | Sticky switcher; one active catalog’s tree in the scroll area. |
| Computed / iterator markers | Yellow Zap or colored `SquareFunction` (and similar) where kind icons do not apply — **not** KindBadge letters. |

## Visual principles

1. **Color carries kind** — table, view, snapshot, replica, and folder are distinguishable without reading labels.
2. **Yellow = Pixeltable accent** — folders, primary CTAs, computed emphasis. Do not strip to gray “for cleanliness.”
3. **Status is green when healthy** — connection dot stays emerald (or equivalent), not muted gray.
4. **Density over decoration** — show `vN` when a version exists; keep error marks; avoid chip piles and dual kind cues (no icon + KindBadge).
5. **Contrast without desaturation** — keep the WCAG AA `--muted-foreground` bump from #1494; do not use a11y as a reason to remove semantic color.
6. **One active catalog context** — switcher is sticky chrome; the scroll region is only that catalog’s directory tree.

## Information architecture

```
Header (logo, version, home)
Search / Lineage
Catalog switcher (Local | cloud URIs | Add cloud…)   ← sticky
Directory tree (active catalog only)                 ← scroll
Collapse                                             ← sticky footer
```

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

## PR #1494 accept / reject / adapt

### Accept (fits this design)

- Multi-catalog browse, `pxt-catalogs` persistence, lazy fetch of hosted trees
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
- Removal of colored type / kind / func accents across schema, cells, and pipeline
- Desaturated connection status
- Removing sidebar `vN` when version exists
- `+ Add catalog` inside the scrolling tree / stacked multi-root forest as default chrome

### Adapt

- Catalog UX → sticky switcher, one active tree (`CatalogSwitcher` + active-catalog fetch)
- Schema header → keep #1494’s cleaner Info model; restore type color/icons on badges
- Computed / iterator → yellow Zap or colored `SquareFunction`, not KindBadge letters

## Component ownership (intended look)

| Concern | Source of truth |
|---------|-----------------|
| Tree icons + `vN` | [`src/components/DirectoryTree.tsx`](src/components/DirectoryTree.tsx) — colored `getNodeIcon`; `vN` when version exists; no KindBadge |
| Catalog switcher | [`src/components/CatalogSwitcher.tsx`](src/components/CatalogSwitcher.tsx) |
| Catalog persistence | [`src/lib/catalogs.ts`](src/lib/catalogs.ts) |
| Column type colors | [`src/lib/column-types.tsx`](src/lib/column-types.tsx) |
| Func accents | [`src/lib/func-styles.ts`](src/lib/func-styles.ts) |
| Shell / routing | [`src/App.tsx`](src/App.tsx) |

The PyPI 0.7.1 bundled SPA may still show KindBadge + in-tree Add catalog. Treat that as divergent until the wheel is rebuilt from sources that follow this doc.

## Future work

Visual restore (colored Lucide + `vN`, KindBadge removed, type/func accents, green status, sticky switcher) is in progress on the local-ui draft. Remaining polish:

1. Keep refining light/dark surface ladder and dense data-toolbar wrap at mid widths.
2. Never reintroduce KindBadge as the primary kind language.
3. Never reintroduce `+ Add catalog` inside the scrolling tree; keep the sticky switcher.
4. Ensure sidebar rows show `vN` whenever `version !== null`.
